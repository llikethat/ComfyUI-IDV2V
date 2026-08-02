"""
Locate the ID-V2V repository and make `idv2v` + the bundled `diffsynth` fork
importable inside the ComfyUI process.

Search order for the repo root:
  1. env var IDV2V_REPO
  2. <this custom node dir>/ID-V2V        (recommended: clone it right here)
  3. <ComfyUI root>/ID-V2V                (sibling of custom_nodes/)
  4. <ComfyUI models dir>/idv2v/ID-V2V

IMPORTANT: the ID-V2V repo ships its OWN diffsynth fork (diffsynth_studio/)
with the custom multi-VACE + SVI Wan pipeline. If a stock DiffSynth-Studio is
already importable in this environment it will NOT contain
`wan_video_new_multiVace_svi` — we therefore *prepend* the fork to sys.path
and verify the right module resolves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT: Path | None = None


def _candidate_paths():
    here = Path(__file__).resolve().parent
    env = os.environ.get("IDV2V_REPO", "").strip()
    if env:
        yield Path(env).expanduser()
    yield here / "ID-V2V"
    # custom_nodes/ComfyUI-IDV2V -> ComfyUI root is two levels up
    comfy_root = here.parent.parent
    yield comfy_root / "ID-V2V"
    try:
        import folder_paths  # ComfyUI runtime module

        yield Path(folder_paths.models_dir) / "idv2v" / "ID-V2V"
    except Exception:
        pass


def _looks_like_repo(p: Path) -> bool:
    return (p / "src" / "idv2v" / "__init__.py").is_file() and (
        p / "diffsynth_studio" / "diffsynth" / "__init__.py"
    ).is_file()


def find_repo_root() -> Path:
    global _REPO_ROOT
    if _REPO_ROOT is not None:
        return _REPO_ROOT
    tried = []
    for cand in _candidate_paths():
        tried.append(str(cand))
        if _looks_like_repo(cand):
            _REPO_ROOT = cand
            return cand
    raise RuntimeError(
        "ComfyUI-IDV2V: could not find the ID-V2V repository. Clone it with\n"
        "    git clone https://github.com/Eyeline-Labs/ID-V2V\n"
        "into the ComfyUI-IDV2V custom node folder (recommended), or set the\n"
        "IDV2V_REPO environment variable to the repo root.\n"
        f"Paths tried: {tried}"
    )


def ensure_importable(check_diffsynth: bool = False) -> Path:
    """Prepend <repo>/src and <repo>/diffsynth_studio to sys.path (idempotent).

    check_diffsynth=True additionally imports `diffsynth` and verifies the
    repo's fork wins over any pip-installed DiffSynth-Studio (the loader node
    uses this; lighter nodes skip the heavy import).
    """
    root = find_repo_root()
    for sub in ("src", "diffsynth_studio"):
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    if check_diffsynth:
        import importlib

        diffsynth = importlib.import_module("diffsynth")
        got = Path(diffsynth.__file__).resolve()
        want_prefix = (root / "diffsynth_studio").resolve()
        if want_prefix not in got.parents:
            raise RuntimeError(
                "ComfyUI-IDV2V: a different `diffsynth` package shadows the ID-V2V "
                f"fork (imported from {got}). The ID-V2V pipeline requires the fork "
                f"bundled at {want_prefix}. Uninstall the conflicting package "
                "(`pip uninstall diffsynth`) or run ComfyUI in a clean venv."
            )
    return root


def default_checkpoints_dir() -> Path:
    """<repo>/checkpoints — where scripts/download_checkpoints.sh puts weights."""
    return find_repo_root() / "checkpoints"
