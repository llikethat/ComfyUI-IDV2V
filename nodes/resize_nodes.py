"""Aspect-preserving batch rescale for high-resolution (2K/4K) sources.

ID-V2V generates at 720p; running SAM3 / building conditions at native 4K
wastes VRAM and RAM for zero quality gain (everything is center-cropped and
resized to the sampler's W x H anyway). Put this node right after your video
loader to bring the working resolution down while keeping the aspect ratio.

Processes in mini-batches so a long 4K IMAGE batch doesn't double peak RAM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_INTERP = {
    "bicubic": dict(mode="bicubic", align_corners=False, antialias=True),
    "bilinear": dict(mode="bilinear", align_corners=False, antialias=True),
    "area": dict(mode="area"),
    "nearest": dict(mode="nearest"),
}


def compute_target_size(w: int, h: int, mode: str, size: int, divisible_by: int, allow_upscale: bool):
    if mode == "long_edge":
        scale = size / max(w, h)
    elif mode == "short_edge":
        scale = size / min(w, h)
    elif mode == "width":
        scale = size / w
    else:  # height
        scale = size / h
    if not allow_upscale:
        scale = min(scale, 1.0)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    d = max(1, divisible_by)
    tw = max(d, round(tw / d) * d)
    th = max(d, round(th / d) * d)
    return tw, th


class IDV2V_AspectResize:
    CATEGORY = "IDV2V/preprocess"
    FUNCTION = "resize"
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("images", "width", "height")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["long_edge", "short_edge", "width", "height"], {"default": "long_edge"}),
                "size": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 2, "tooltip": "Target size for the chosen edge. 1280 long-edge matches the 720p generation width."}),
                "divisible_by": ("INT", {"default": 2, "min": 1, "max": 64, "tooltip": "Round target W/H to a multiple of this (2 keeps video encoders happy)."}),
                "interpolation": (list(_INTERP.keys()), {"default": "bicubic"}),
                "allow_upscale": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 16, "min": 1, "max": 256, "tooltip": "Frames resized per mini-batch (limits peak RAM on long 4K clips)."}),
            }
        }

    def resize(self, images: torch.Tensor, mode: str, size: int, divisible_by: int,
               interpolation: str, allow_upscale: bool, batch_size: int):
        b, h, w, c = images.shape
        tw, th = compute_target_size(w, h, mode, size, divisible_by, allow_upscale)
        if (tw, th) == (w, h):
            return (images, w, h)

        kw = _INTERP[interpolation]
        chunks = []
        for i in range(0, b, batch_size):
            x = images[i : i + batch_size].permute(0, 3, 1, 2)  # BHWC -> BCHW
            y = F.interpolate(x, size=(th, tw), **kw)
            chunks.append(y.clamp(0, 1).permute(0, 2, 3, 1).contiguous())
        out = torch.cat(chunks, dim=0)
        print(f"[IDV2V] AspectResize: {w}x{h} -> {tw}x{th} ({b} frames)")
        return (out, tw, th)
