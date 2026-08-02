"""Shared helpers: ComfyUI tensor <-> PIL conversion and spatial transforms.

ComfyUI conventions:
  IMAGE = torch.float32 tensor [B, H, W, C] in [0, 1]
  MASK  = torch.float32 tensor [B, H, W]    in [0, 1]
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image


def images_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """IMAGE batch [B,H,W,C] -> list of PIL RGB frames."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    arr = (images.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return [Image.fromarray(a[..., :3], mode="RGB") for a in arr]


def iter_images_as_pil(images: torch.Tensor):
    """Yield PIL RGB frames one at a time from an IMAGE batch.

    Unlike images_to_pil, this never materializes the whole batch as uint8 at
    once — with an optional per-frame transform applied by the caller it keeps
    peak RAM flat on long / high-resolution clips.
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    for i in range(images.shape[0]):
        arr = (images[i].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        yield Image.fromarray(arr[..., :3], mode="RGB")


def pil_to_images(frames: List[Image.Image]) -> torch.Tensor:
    """List of PIL RGB frames -> IMAGE batch [B,H,W,C] float32 in [0,1]."""
    arrs = [np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0 for f in frames]
    return torch.from_numpy(np.stack(arrs, axis=0))


def masks_to_bool_np(masks: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """MASK batch [B,H,W] (or [H,W]) -> bool ndarray [B,H,W]."""
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    return (masks > threshold).cpu().numpy()


def bool_np_to_masks(masks: np.ndarray) -> torch.Tensor:
    """bool ndarray [B,H,W] -> MASK float tensor [B,H,W]."""
    return torch.from_numpy(masks.astype(np.float32))


def center_crop_and_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Match ID-V2V pipeline.py: resize to target aspect first, then center-crop."""
    w, h = img.size
    target_aspect = height / width
    aspect = h / w

    if (h == height) and (w == width):
        return img
    if abs(aspect - target_aspect) < 1e-6:
        return img.resize((width, height), Image.BICUBIC)

    if aspect > target_aspect:  # too tall -> match width, crop height
        new_w = width
        new_h = int(aspect * new_w)
    else:  # too wide -> match height, crop width
        new_h = height
        new_w = int(new_h / aspect)
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    rw, rh = resized.size
    left = (rw - width) // 2
    top = (rh - height) // 2
    return resized.crop((left, top, left + width, top + height))


def compute_clip_schedule(total_frames: int, num_frames_per_clip: int):
    """Identical to idv2v.inference.pipeline.compute_clip_schedule.

    Regular clips advance by stride = num_frames_per_clip - 1 (1-frame overlap);
    the last clip is anchored at the end so it is always full-length.
    """
    if total_frames <= num_frames_per_clip:
        return [(0, total_frames)]
    clips = []
    start = 0
    stride = num_frames_per_clip - 1
    while start + num_frames_per_clip < total_frames:
        clips.append((start, start + num_frames_per_clip))
        start += stride
    clips.append((total_frames - num_frames_per_clip, total_frames))
    return clips


def slice_frames(all_frames, start: int, end: int):
    """frames[start:end], padded by repeating the last frame if end > len."""
    n = len(all_frames)
    if end <= n:
        return all_frames[start:end]
    result = list(all_frames[start:n])
    result += [all_frames[-1]] * (end - n)
    return result
