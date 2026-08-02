"""ID-V2V SAM3 video segmentation node.

Replicates scripts/preprocess.sh step 1 (idv2v.preprocess.sam3) fully in
memory: SAM3 Promptable Concept Segmentation with tracking across the video,
followed by "Secret Panda" mask cleanup (hole-fill -> morph close -> bridge
gaps -> hole-fill) per object, and optionally on the union mask.

Long-video / high-resolution handling:
  * chunk_size:            process the video in independent temporal chunks.
                           SAM3 stores the whole session's frames on the
                           inference device, which OOMs on long clips; chunks
                           bound that. The text prompt re-detects the concept
                           in every chunk, and since ID-V2V consumes only the
                           UNION mask, cross-chunk object-ID consistency is
                           irrelevant. (Per-chunk detections can differ
                           slightly at boundaries; if you see mask flicker
                           there, raise chunk_size.)
  * processing_max_edge:   run SAM3 (and Secret Panda) at a downscaled
                           resolution, then nearest-upscale the union masks
                           back to input resolution. Big VRAM/time win on
                           2K/4K sources. For 4K it is usually better still
                           to put IDV2V Aspect Resize BEFORE this node so the
                           whole graph works at ~720p.
  * video_storage_device:  'cpu' keeps session frames in system RAM instead
                           of VRAM (slower, much larger capacity).

Input frames come from any ComfyUI video loader (e.g. VHS Load Video).
Outputs the per-frame union MASK plus an overlay IMAGE for debugging (the
overlay is rendered at processing resolution).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from .utils import bool_np_to_masks, pil_to_images

# key -> (model, processor, device); one entry, replaced when model_path changes
_SAM3_CACHE: dict = {}


def _get_sam3(model_path: str):
    from .. import idv2v_bootstrap

    idv2v_bootstrap.ensure_importable()

    resolved = model_path.strip()
    if not resolved:
        resolved = str(idv2v_bootstrap.default_checkpoints_dir() / "sam3")

    if _SAM3_CACHE.get("key") == resolved:
        return _SAM3_CACHE["value"]

    from idv2v.preprocess.sam3 import init_sam3_video

    model, processor, device = init_sam3_video(model_path=resolved, dtype=torch.bfloat16)
    _SAM3_CACHE["key"] = resolved
    _SAM3_CACHE["value"] = (model, processor, device)
    return model, processor, device


def unload_sam3():
    _SAM3_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_sam3_chunk(model, processor, device, frames, prompt, storage_device):
    """Same as idv2v.preprocess.sam3.run_sam3_video, with selectable frame storage."""
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=device,
        video_storage_device=storage_device,
        dtype=torch.bfloat16,
    )
    session = processor.add_text_prompt(inference_session=session, text=prompt)

    outputs_per_frame = {}
    with torch.inference_mode():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=session,
            max_frame_num_to_track=len(frames) - 1,
        ):
            processed = processor.postprocess_outputs(session, model_outputs)
            outputs_per_frame[int(model_outputs.frame_idx)] = processed
    del session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs_per_frame


class IDV2V_SAM3VideoSegment:
    """SAM3 text-prompted video segmentation + Secret Panda mask cleanup."""

    CATEGORY = "IDV2V/preprocess"
    FUNCTION = "segment"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "overlay")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source video frames (e.g. from VHS Load Video)."}),
                "sam_prompt": ("STRING", {"default": "person", "tooltip": "What to segment: 'person', 'head', 'dog', ..."}),
                "model_path": ("STRING", {"default": "", "tooltip": "Local SAM3 dir or HF repo id. Empty = <ID-V2V repo>/checkpoints/sam3. SAM3 is gated on HF: run `hf auth login` once."}),
                "close_kernel": ("INT", {"default": 10, "min": 0, "max": 64, "tooltip": "Secret Panda morphological-close kernel."}),
                "bridge_distance": ("INT", {"default": 15, "min": 0, "max": 128, "tooltip": "Secret Panda gap-bridging distance."}),
                "joint_mask_post_proc": ("BOOLEAN", {"default": True, "tooltip": "Also run Secret Panda on the union mask (fills slivers between adjacent people)."}),
                "chunk_size": ("INT", {"default": 0, "min": 0, "max": 4096, "tooltip": "0 = whole video in one SAM3 session. >0 = process in independent temporal chunks of this many frames (bounds VRAM/RAM on long videos; e.g. 150-300)."}),
                "processing_max_edge": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 2, "tooltip": "0 = segment at input resolution. >0 = downscale frames so the long edge <= this for SAM3, then nearest-upscale masks back to input resolution (e.g. 1280 for 2K/4K sources)."}),
                "video_storage_device": (["gpu", "cpu"], {"default": "gpu", "tooltip": "Where the SAM3 session stores video frames. 'cpu' trades speed for capacity on long chunks."}),
                "unload_after": ("BOOLEAN", {"default": True, "tooltip": "Free SAM3 from VRAM after this node (recommended before loading the 14B Wan pipeline)."}),
            }
        }

    def segment(
        self,
        images: torch.Tensor,
        sam_prompt: str,
        model_path: str,
        close_kernel: int,
        bridge_distance: int,
        joint_mask_post_proc: bool,
        chunk_size: int,
        processing_max_edge: int,
        video_storage_device: str,
        unload_after: bool,
    ):
        from .. import idv2v_bootstrap

        idv2v_bootstrap.ensure_importable()
        from PIL import Image

        from idv2v.preprocess.sam3 import overlay_instances, pack_per_frame_instances
        from idv2v.preprocess.secret_panda import secret_panda

        n, in_h, in_w, _ = images.shape

        # Processing resolution (aspect-preserving long-edge cap)
        if processing_max_edge > 0 and max(in_w, in_h) > processing_max_edge:
            scale = processing_max_edge / max(in_w, in_h)
            proc_w = max(2, round(in_w * scale / 2) * 2)
            proc_h = max(2, round(in_h * scale / 2) * 2)
        else:
            proc_w, proc_h = in_w, in_h
        rescaled = (proc_w, proc_h) != (in_w, in_h)
        if rescaled:
            print(f"[IDV2V] SAM3 processing at {proc_w}x{proc_h} (input {in_w}x{in_h}); masks upscaled back to input resolution")

        def frame_to_pil(i: int) -> Image.Image:
            arr = (images[i].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            pil = Image.fromarray(arr[..., :3], mode="RGB")
            if rescaled:
                pil = pil.resize((proc_w, proc_h), Image.BICUBIC)
            return pil

        model, processor, device = _get_sam3(model_path)
        storage_device = torch.device("cpu") if video_storage_device == "cpu" else device

        union_masks = np.zeros((n, in_h, in_w), dtype=bool)
        overlay_frames: List[Image.Image] = []

        step = chunk_size if chunk_size > 0 else n
        for chunk_start in range(0, n, step):
            chunk_end = min(chunk_start + step, n)
            chunk_frames = [frame_to_pil(i) for i in range(chunk_start, chunk_end)]
            if chunk_size > 0:
                print(f"[IDV2V] SAM3 chunk [{chunk_start}, {chunk_end}) / {n}")

            outputs_per_frame = _run_sam3_chunk(
                model, processor, device, chunk_frames, sam_prompt, storage_device
            )
            per_frame_instances = pack_per_frame_instances(outputs_per_frame)

            for local_idx in range(len(chunk_frames)):
                global_idx = chunk_start + local_idx
                instances = sorted(per_frame_instances.get(local_idx, []), key=lambda d: int(d["id"]))
                union = np.zeros((proc_h, proc_w), dtype=bool)
                for inst in instances:
                    m = inst["mask"]
                    if m.dtype != np.bool_:
                        m = m > 0.5
                    m = np.asarray(m, dtype=bool)
                    if m.shape != (proc_h, proc_w):
                        raise ValueError(
                            f"SAM3 mask shape {m.shape} != processing shape {(proc_h, proc_w)} at frame {global_idx}"
                        )
                    m = secret_panda(
                        m.astype(np.uint8),
                        fill_holes_first=True,
                        close_kernel=close_kernel,
                        bridge_distance=bridge_distance,
                    ).astype(bool)
                    inst["mask"] = m  # so the overlay shows post-processed masks
                    union |= m
                if joint_mask_post_proc and union.any():
                    union = secret_panda(
                        union.astype(np.uint8),
                        fill_holes_first=True,
                        close_kernel=close_kernel,
                        bridge_distance=bridge_distance,
                    ).astype(bool)

                if rescaled:
                    union_img = Image.fromarray(union.astype(np.uint8) * 255, mode="L")
                    union = np.asarray(union_img.resize((in_w, in_h), Image.NEAREST)) > 0
                union_masks[global_idx] = union

                overlay_frames.append(overlay_instances(chunk_frames[local_idx], instances))

            del outputs_per_frame, per_frame_instances, chunk_frames

        if unload_after:
            unload_sam3()

        return (bool_np_to_masks(union_masks), pil_to_images(overlay_frames))
