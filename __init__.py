"""
ComfyUI-IDV2V
=============
ComfyUI custom nodes for Eyeline Labs' ID-V2V
(identity-preserving video-to-video restylization / relighting,
https://github.com/Eyeline-Labs/ID-V2V).

Video / image loading and saving is intentionally NOT provided here — use core
ComfyUI nodes or ComfyUI-VideoHelperSuite (VHS) "Load Video" / "Video Combine".

The heavy dependencies (the ID-V2V repo itself + its bundled diffsynth fork)
are resolved lazily at node-execution time via `idv2v_bootstrap`, so this pack
always loads in ComfyUI even before the repo is installed, and gives a clear
error message when a node actually runs without it.
"""

from .nodes.sam3_nodes import IDV2V_SAM3VideoSegment
from .nodes.condition_nodes import IDV2V_ForegroundOnGray
from .nodes.resize_nodes import IDV2V_AspectResize
from .nodes.mask_nodes import IDV2V_SecretPandaCleanup
from .nodes.loader_nodes import IDV2V_LoadPipeline
from .nodes.sampler_nodes import IDV2V_Keyframe, IDV2V_Sampler
from .nodes.viz_nodes import IDV2V_FlipTest

NODE_CLASS_MAPPINGS = {
    "IDV2V_SAM3VideoSegment": IDV2V_SAM3VideoSegment,
    "IDV2V_AspectResize": IDV2V_AspectResize,
    "IDV2V_SecretPandaCleanup": IDV2V_SecretPandaCleanup,
    "IDV2V_ForegroundOnGray": IDV2V_ForegroundOnGray,
    "IDV2V_LoadPipeline": IDV2V_LoadPipeline,
    "IDV2V_Keyframe": IDV2V_Keyframe,
    "IDV2V_Sampler": IDV2V_Sampler,
    "IDV2V_FlipTest": IDV2V_FlipTest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IDV2V_SAM3VideoSegment": "ID-V2V SAM3 Video Segmenter (chunked)",
    "IDV2V_AspectResize": "ID-V2V Aspect Resize (2K/4K)",
    "IDV2V_SecretPandaCleanup": "ID-V2V Secret Panda Mask Cleanup",
    "IDV2V_ForegroundOnGray": "ID-V2V Foreground-on-Gray Condition",
    "IDV2V_LoadPipeline": "ID-V2V Pipeline Loader (Wan2.1 + VACE)",
    "IDV2V_Keyframe": "ID-V2V Keyframe",
    "IDV2V_Sampler": "ID-V2V Sampler (multi-clip)",
    "IDV2V_FlipTest": "ID-V2V Flip Test (A/B viz)",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
