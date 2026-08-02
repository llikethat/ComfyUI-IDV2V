"""ID-V2V foreground-on-gray condition node.

Replicates scripts/preprocess.sh step 2 (idv2v.preprocess.orig_pixel): keep
pixels inside the per-frame mask, fill everything else with gray (127). This
is the single VACE control signal the default `idv2v` model consumes.

Note: for the RELIGHTING use case, skip this node entirely and feed the raw
source frames straight into the sampler's `condition_0` input.
"""

from __future__ import annotations

import torch


class IDV2V_ForegroundOnGray:
    CATEGORY = "IDV2V/preprocess"
    FUNCTION = "compose"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("condition",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source video frames."}),
                "masks": ("MASK", {"tooltip": "Per-frame masks from the SAM3 node (same frame count and resolution)."}),
                "gray_value": ("INT", {"default": 127, "min": 0, "max": 255}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "resize_masks_if_needed": ("BOOLEAN", {"default": False, "tooltip": "Nearest-resize masks to the image resolution when they differ (e.g. masks from an external chunked SAM3 node run at a different resolution). Upstream orig_pixel.py refuses to resize, hence default off."}),
            }
        }

    def compose(self, images: torch.Tensor, masks: torch.Tensor, gray_value: int,
                mask_threshold: float, resize_masks_if_needed: bool = False):
        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        if images.shape[0] != masks.shape[0]:
            raise ValueError(
                f"Frame count mismatch: images={images.shape[0]}, masks={masks.shape[0]}. "
                "Masks must come from the same video."
            )
        if images.shape[1:3] != masks.shape[1:3] and resize_masks_if_needed:
            import torch.nn.functional as F
            masks = F.interpolate(
                masks.unsqueeze(1), size=(images.shape[1], images.shape[2]), mode="nearest"
            ).squeeze(1)
        if images.shape[1:3] != masks.shape[1:3]:
            raise ValueError(
                f"Resolution mismatch: images={tuple(images.shape[1:3])}, masks={tuple(masks.shape[1:3])}. "
                "ID-V2V applies masks at native resolution (no resizing), matching orig_pixel.py."
            )

        keep = (masks > mask_threshold).unsqueeze(-1)  # [B,H,W,1]
        gray = torch.full_like(images, float(gray_value) / 255.0)
        out = torch.where(keep, images, gray)
        return (out,)
