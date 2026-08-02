"""ID-V2V sampler: clip-by-clip I2V + VACE multi-control generation.

Faithful in-memory port of the generation loop in idv2v/inference/pipeline.py:
  * center-crop + resize of conditions / first frame / keyframes to (W, H)
  * keyframe injection into ALL condition videos (index 0 reserved for the
    stylized first frame) + VACE mask (white=reactive, black=hard-pinned)
  * truncation to max_num_frames, then the multi-clip schedule
    (stride = num_frames_per_clip - 1, last clip anchored at the end)
  * per-clip chaining via the previous clip's splice frame as the I2V anchor
  * anti-drift (SVI) padding with the stylized first frame (ref_pad_num)
  * stitching (on overlaps, the LATER clip's frames win)

Single-GPU (no torchrun/USP — ComfyUI is a single process).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from .utils import (
    center_crop_and_resize,
    compute_clip_schedule,
    images_to_pil,
    iter_images_as_pil,
    pil_to_images,
    slice_frames,
)

# Wan 2.1's standard Chinese quality-degradation negative prompt (repo default).
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class IDV2V_Keyframe:
    """Attach a stylized keyframe at a 0-based frame index (>= 1). Chainable."""

    CATEGORY = "IDV2V"
    FUNCTION = "build"
    RETURN_TYPES = ("IDV2V_KEYFRAMES",)
    RETURN_NAMES = ("keyframes",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Stylized keyframe image (first image of the batch is used)."}),
                "frame_index": ("INT", {"default": 80, "min": 1, "max": 100000, "tooltip": "0-based frame index to pin. Index 0 is reserved for the stylized first frame."}),
            },
            "optional": {
                "keyframes": ("IDV2V_KEYFRAMES", {"tooltip": "Chain from another IDV2V Keyframe node to pin multiple frames."}),
            },
        }

    def build(self, image: torch.Tensor, frame_index: int, keyframes=None):
        pil = images_to_pil(image)[0]
        entries = list(keyframes) if keyframes else []
        if any(idx == frame_index for _, idx in entries):
            raise ValueError(f"Duplicate keyframe index {frame_index}.")
        entries.append((pil, int(frame_index)))
        entries.sort(key=lambda e: e[1])
        return (entries,)


class IDV2V_Sampler:
    CATEGORY = "IDV2V"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("IDV2V_PIPE",),
                "condition_0": ("IMAGE", {"tooltip": "VACE condition video frames. Default idv2v model: foreground-on-gray. Relighting: the raw source frames."}),
                "first_frame": ("IMAGE", {"tooltip": "Stylized first frame — the look you want (frame 0). May be imperfect / non-aligned."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Empty = Wan 2.1's default Chinese quality-degradation prompt."}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 16}),
                "num_frames_per_clip": ("INT", {"default": 81, "min": 5, "max": 241, "step": 4, "tooltip": "Must be 1 + a multiple of 4 (45, 49, 81, 121, ...)."}),
                "max_num_frames": ("INT", {"default": 240, "min": 0, "max": 100000, "tooltip": "Cap on total output frames (0 = no cap). Longer conditions are generated clip-by-clip."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "vace_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "ref_pad_num": ("INT", {"default": -1, "min": -1, "max": 240, "tooltip": "-1 = full anti-drift (SVI) padding with the first frame; 0 = original VACE zero padding; N = N-frame padding."}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 2**31 - 1}),
                "different_seed_per_clip": ("BOOLEAN", {"default": False, "tooltip": "Each clip uses seed + 42*clip_idx."}),
            },
            "optional": {
                "keyframes": ("IDV2V_KEYFRAMES", {"tooltip": "Extra frame-level anchors from IDV2V Keyframe node(s)."}),
                "condition_1": ("IMAGE", {"tooltip": "2nd VACE condition (idv2v_with_normal_depth: surface normals). Leave unconnected for the default model."}),
                "condition_2": ("IMAGE", {"tooltip": "3rd VACE condition (idv2v_with_normal_depth: depth). Leave unconnected for the default model."}),
            },
        }

    def generate(
        self,
        pipe,
        condition_0: torch.Tensor,
        first_frame: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        num_frames_per_clip: int,
        max_num_frames: int,
        steps: int,
        cfg_scale: float,
        vace_scale: float,
        ref_pad_num: int,
        seed: int,
        different_seed_per_clip: bool,
        keyframes=None,
        condition_1: Optional[torch.Tensor] = None,
        condition_2: Optional[torch.Tensor] = None,
    ):
        if (num_frames_per_clip - 1) % 4 != 0:
            raise ValueError(
                f"num_frames_per_clip must be 1 plus a multiple of 4 (e.g. 45, 49, 81, 121); got "
                f"{num_frames_per_clip}. The Wan VACE engine rounds to the nearest 4k+1, which would "
                "desync the multi-clip schedule."
            )
        if not prompt.strip():
            raise ValueError("Prompt is empty. ID-V2V requires a text prompt (the repo reads it from prompt.txt).")

        negative = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT

        # --- Prepare inputs (mirrors pipeline.py Step 0) ---
        input_image = center_crop_and_resize(images_to_pil(first_frame)[0], width, height)

        # Stream conversion: each source frame is converted to PIL and immediately
        # center-cropped to (W, H), so a long 2K/4K condition batch never exists as
        # a full-resolution PIL list in RAM.
        cond_batches = [c for c in (condition_0, condition_1, condition_2) if c is not None]
        frames_conditions: List[List[Image.Image]] = [
            [center_crop_and_resize(f, width, height) for f in iter_images_as_pil(c)]
            for c in cond_batches
        ]
        counts = [len(fc) for fc in frames_conditions]
        if len(set(counts)) != 1:
            raise ValueError(f"All condition inputs must have the same frame count; got {counts}.")
        total_frames = counts[0]

        # Keyframe injection (before truncation; indices refer to the full video)
        kf_indices: List[int] = []
        if keyframes:
            effective_max = total_frames if max_num_frames <= 0 else min(total_frames, max_num_frames)
            cond_w, cond_h = frames_conditions[0][0].size
            cond_ar = cond_w / cond_h
            for i, (kf_img, kf_idx) in enumerate(keyframes):
                kf_ar = kf_img.width / kf_img.height
                if abs(kf_ar - cond_ar) > 0.1:
                    raise ValueError(
                        f"Keyframe {i} aspect ratio ({kf_ar:.4f}) differs from the condition video "
                        f"({cond_ar:.4f}) by more than 0.1."
                    )
                if kf_idx >= effective_max:
                    raise ValueError(
                        f"Keyframe index {kf_idx} >= effective output length {effective_max} "
                        f"(min of total_frames={total_frames} and max_num_frames={max_num_frames})."
                    )
                resized_kf = center_crop_and_resize(kf_img, width, height)
                for c in range(len(frames_conditions)):
                    frames_conditions[c][kf_idx] = resized_kf
                kf_indices.append(kf_idx)
            print(f"[IDV2V] Injected {len(kf_indices)} keyframes at indices {kf_indices}")

        # Truncate to max_num_frames
        if max_num_frames > 0 and total_frames > max_num_frames:
            print(f"[IDV2V] Truncating from {total_frames} to max_num_frames={max_num_frames}")
            frames_conditions = [fc[:max_num_frames] for fc in frames_conditions]
            total_frames = max_num_frames

        clip_schedule = compute_clip_schedule(total_frames, num_frames_per_clip)
        num_clips = len(clip_schedule)
        print(f"[IDV2V] Clip schedule ({num_clips} clips, num_frames_per_clip={num_frames_per_clip}):")
        for i, (s, e) in enumerate(clip_schedule):
            overlap = clip_schedule[i - 1][1] - s if i > 0 else 0
            print(f"  Clip {i}: frames [{s}, {e}) = {e - s} frames" + (f", overlap={overlap}" if overlap else ""))

        # VACE mask: white=reactive everywhere, black=hard-pin at keyframe indices
        white = Image.new("RGB", (width, height), (255, 255, 255))
        black = Image.new("RGB", (width, height), (0, 0, 0))
        kf_set = set(kf_indices)
        full_mask_frames = [black if i in kf_set else white for i in range(total_frames)]

        # --- Multi-clip generation loop (mirrors pipeline.py Step 4) ---
        all_clips: List[List[Image.Image]] = []
        current_input_image = input_image

        for clip_idx, (frame_start, frame_end) in enumerate(clip_schedule):
            clip_seed = (seed + 42 * clip_idx) if different_seed_per_clip else seed
            print(f"[IDV2V] --- Clip {clip_idx + 1}/{num_clips} frames=[{frame_start}, {frame_end}) seed={clip_seed} ---")

            if clip_idx > 0:
                splice_idx = clip_schedule[clip_idx][0] - clip_schedule[clip_idx - 1][0]
                current_input_image = all_clips[-1][splice_idx]

            clip_end = frame_start + num_frames_per_clip
            clip_conditions = [slice_frames(cond, frame_start, clip_end) for cond in frames_conditions]
            clip_mask_single = slice_frames(full_mask_frames, frame_start, clip_end)
            clip_mask = [clip_mask_single] * len(frames_conditions)

            generated = pipe(
                prompt=prompt,
                negative_prompt=negative,
                input_image=current_input_image,
                random_ref_frame=input_image,  # always the user's stylized first frame for SVI padding
                ref_pad_num=ref_pad_num,
                vace_video=clip_conditions,
                vace_video_mask=clip_mask,
                seed=clip_seed,
                num_inference_steps=steps,
                use_multi_control_vace=True,
                height=height,
                width=width,
                num_frames=num_frames_per_clip,
                cfg_scale=cfg_scale,
                tiled=False,
                vace_scale=vace_scale,
            )
            all_clips.append(list(generated))
            print(f"[IDV2V]   Clip {clip_idx + 1}: {len(generated)} frames generated")

        # --- Stitch (on overlaps, the later clip's frames win) ---
        if num_clips == 1:
            combined = all_clips[0]
            if total_frames < num_frames_per_clip:
                combined = combined[:total_frames]
        else:
            combined = list(all_clips[0])
            for i in range(1, num_clips):
                overlap = clip_schedule[i - 1][1] - clip_schedule[i][0]
                combined = combined[:-overlap] + list(all_clips[i])
            print(f"[IDV2V] Stitched {num_clips} clips into {len(combined)} frames")

        return (pil_to_images(combined),)
