"""ID-V2V flip-test node.

Builds the repo's headline A/B visualization (idv2v/inference/flip_test.py) —
plays the generated video and periodically freezes, alternating
Generated (green pill) <-> Source (red pill) — but returns the frames as an
IMAGE batch so encoding stays with your video-save node (e.g. VHS Video
Combine). Reuses the repo's own stamping helpers for identical visuals.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image

from .utils import images_to_pil, pil_to_images


class IDV2V_FlipTest:
    CATEGORY = "IDV2V/viz"
    FUNCTION = "build"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("flip_frames",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "source": ("IMAGE", {"tooltip": "Source frames, resized/cropped to match the generated frames (feed them through the sampler's resolution, or use an ImageScale/crop node)."}),
                "output_fps": ("INT", {"default": 16, "min": 1, "max": 120, "tooltip": "FPS you will encode at — used to size the freeze holds."}),
                "play_block_frames": ("INT", {"default": 20, "min": 1, "max": 500}),
                "flip_pairs_per_freeze": ("INT", {"default": 2, "min": 1, "max": 10}),
                "flip_hold_sec": ("FLOAT", {"default": 0.75, "min": 0.1, "max": 5.0, "step": 0.05}),
                "max_flips": ("INT", {"default": 5, "min": 1, "max": 50}),
                "show_border_on_freeze": ("BOOLEAN", {"default": True}),
            }
        }

    def build(
        self,
        generated: torch.Tensor,
        source: torch.Tensor,
        output_fps: int,
        play_block_frames: int,
        flip_pairs_per_freeze: int,
        flip_hold_sec: float,
        max_flips: int,
        show_border_on_freeze: bool,
    ):
        from .. import idv2v_bootstrap

        idv2v_bootstrap.ensure_importable()
        from idv2v.inference.flip_test import PILL_FONT_SIZE, _load_font, _stamp_gen, _stamp_src

        frames_gen = images_to_pil(generated)
        frames_src = images_to_pil(source)
        n = min(len(frames_gen), len(frames_src))
        frames_gen, frames_src = frames_gen[:n], frames_src[:n]
        if n == 0:
            raise ValueError("No frames to compare.")
        if frames_gen[0].size != frames_src[0].size:
            raise ValueError(
                f"Resolution mismatch: generated={frames_gen[0].size}, source={frames_src[0].size}. "
                "Resize/crop the source to the generated resolution first."
            )

        # Frame assembly identical to build_flip_test, minus the mp4 write.
        hold_n = max(1, int(round(flip_hold_sec * output_fps)))
        font = _load_font(PILL_FONT_SIZE)
        out: List[Image.Image] = []

        total_blocks = (n + play_block_frames - 1) // play_block_frames
        if total_blocks <= max_flips:
            freeze_block_set = set(range(total_blocks))
        else:
            idxs = np.linspace(0, total_blocks - 1, max_flips)
            freeze_block_set = {int(round(i)) for i in idxs}

        pos = 0
        block_idx = 0
        while pos < n:
            block_end = min(pos + play_block_frames, n)
            for i in range(pos, block_end):
                out.append(_stamp_gen(frames_gen[i], font, with_border=False))
            if block_idx in freeze_block_set:
                freeze_idx = block_end - 1
                g_frozen = _stamp_gen(frames_gen[freeze_idx], font, with_border=show_border_on_freeze)
                s_frozen = _stamp_src(frames_src[freeze_idx], font, with_border=show_border_on_freeze)
                for _ in range(flip_pairs_per_freeze):
                    out.extend([g_frozen] * hold_n)
                    out.extend([s_frozen] * hold_n)
            pos = block_end
            block_idx += 1

        return (pil_to_images(out),)
