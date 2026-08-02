"""Standalone Secret Panda mask cleanup.

ID-V2V's mask post-processing (hole-fill -> morphological close -> bridge
gaps -> hole-fill) exposed as a MASK -> MASK node, so masks produced by ANY
external segmentation node — e.g. a chunked SAM3 node from another pack —
can be brought up to what the ID-V2V pipeline expects before
ForegroundOnGray. The only contract downstream is: one mask per frame, at
the same resolution as the frames it will be applied to.
"""

from __future__ import annotations

import numpy as np
import torch

from .utils import bool_np_to_masks, masks_to_bool_np


class IDV2V_SecretPandaCleanup:
    CATEGORY = "IDV2V/preprocess"
    FUNCTION = "clean"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {"tooltip": "Per-frame masks from any segmentation node (external chunked SAM3 nodes included)."}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "close_kernel": ("INT", {"default": 10, "min": 0, "max": 64}),
                "bridge_distance": ("INT", {"default": 15, "min": 0, "max": 128}),
            }
        }

    def clean(self, masks: torch.Tensor, mask_threshold: float, close_kernel: int, bridge_distance: int):
        from .. import idv2v_bootstrap

        idv2v_bootstrap.ensure_importable()
        from idv2v.preprocess.secret_panda import secret_panda

        arr = masks_to_bool_np(masks, threshold=mask_threshold)
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            if arr[i].any():
                out[i] = secret_panda(
                    arr[i].astype(np.uint8),
                    fill_holes_first=True,
                    close_kernel=close_kernel,
                    bridge_distance=bridge_distance,
                ).astype(bool)
            else:
                out[i] = arr[i]
        return (bool_np_to_masks(out),)
