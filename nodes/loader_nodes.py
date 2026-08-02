"""ID-V2V pipeline loader.

Wraps the model-loading portion of idv2v/inference/pipeline.py:
  * WanVideoPipeline.from_pretrained with T5 + VAE + CLIP (I2V) configs
  * fast-load of the finetuned idv2v .pth (empty DiT + VACE, mmap'd weights)
    — or, without a finetuned checkpoint, base I2V DiT + selective VACE-14B
  * enable_vram_management with CPU offload (single-GPU path; ComfyUI is a
    single process, so the repo's torchrun/USP multi-GPU mode does not apply)

The loaded pipeline is cached across queue runs and only reloaded when the
checkpoint / model dir / settings change.

Expected `wan_model_dir` layout is exactly what scripts/download_checkpoints.sh
produces (default <ID-V2V repo>/checkpoints/wan):
  Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth
  Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth
  Wan-AI/Wan2.1-T2V-14B/google/*                       (tokenizer)
  Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
  Wan-AI/Wan2.1-I2V-14B-{480,720}P/diffusion_pytorch_model*.safetensors  (base-DiT path only)
  Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors            (base-VACE path only)
"""

from __future__ import annotations

import os

import torch

_PIPE_CACHE: dict = {}


class IDV2V_LoadPipeline:
    CATEGORY = "IDV2V"
    FUNCTION = "load"
    RETURN_TYPES = ("IDV2V_PIPE",)
    RETURN_NAMES = ("pipe",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "idv2v_checkpoint": ("STRING", {"default": "", "tooltip": "Path to the finetuned idv2v.pth (or idv2v_with_normal_depth.pth). Empty = <ID-V2V repo>/checkpoints/idv2v.pth. Set to 'none' to use base Wan I2V + VACE-14B weights instead."}),
                "wan_model_dir": ("STRING", {"default": "", "tooltip": "Dir with Wan-AI/... subfolders (T5, VAE, tokenizer, CLIP). Empty = <ID-V2V repo>/checkpoints/wan."}),
                "i2v_resolution": (["720", "480"], {"default": "720", "tooltip": "Base Wan2.1 I2V variant. Only used when idv2v_checkpoint is 'none' (the finetuned checkpoint replaces the DiT)."}),
                "vram_buffer": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 60.0, "step": 0.5, "tooltip": "VRAM management buffer (GB). Repo default for single-GPU 720p is 10; 5 is usually enough."}),
            }
        }

    def load(self, idv2v_checkpoint: str, wan_model_dir: str, i2v_resolution: str, vram_buffer: float):
        from .. import idv2v_bootstrap

        repo = idv2v_bootstrap.ensure_importable(check_diffsynth=True)
        ckpt_dir = idv2v_bootstrap.default_checkpoints_dir()

        ckpt = idv2v_checkpoint.strip()
        if not ckpt:
            ckpt = str(ckpt_dir / "idv2v.pth")
        use_finetuned = ckpt.lower() != "none"
        if use_finetuned and not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"idv2v checkpoint not found: {ckpt}\n"
                "Run `bash scripts/download_checkpoints.sh` in the ID-V2V repo, "
                "or point idv2v_checkpoint at a local .pth (or set it to 'none' "
                "for base Wan weights)."
            )

        wan_dir = wan_model_dir.strip() or str(ckpt_dir / "wan")
        if not os.path.isdir(wan_dir):
            raise FileNotFoundError(
                f"wan_model_dir not found: {wan_dir}\n"
                "Run `bash scripts/download_checkpoints.sh` in the ID-V2V repo "
                "(fetches T5, VAE, tokenizer and CLIP) or set the correct path."
            )

        cache_key = (ckpt if use_finetuned else "none", wan_dir, i2v_resolution, float(vram_buffer))
        if _PIPE_CACHE.get("key") == cache_key:
            return (_PIPE_CACHE["pipe"],)

        # Drop any previously cached pipeline before loading a new one (14B model).
        if "pipe" in _PIPE_CACHE:
            _PIPE_CACHE.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Import order matters: idv2v.inference.pipeline installs the
        # transformers-5.x compat shims and resolves the diffsynth fork.
        from idv2v.inference import pipeline as idv2v_pipeline
        from diffsynth.pipelines.wan_video_new_multiVace_svi import (
            ModelConfig,
            WanVideoPipeline,
            load_vace_from_checkpoint,
        )

        model_ids = [
            (f"Wan-AI/Wan2.1-I2V-14B-{i2v_resolution}P", "diffusion_pytorch_model*.safetensors"),
            ("Wan-AI/Wan2.1-T2V-14B", "models_t5_umt5-xxl-enc-bf16.pth"),
            ("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth"),
            ("Wan-AI/Wan2.1-I2V-14B-480P", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ]
        model_configs = []
        for model_id, pattern in model_ids:
            if use_finetuned and pattern.startswith("diffusion_pytorch_model"):
                print(f"[IDV2V] Skipping base DiT load (finetuned checkpoint replaces it): {model_id}")
                continue
            model_configs.append(
                ModelConfig(model_id=model_id, origin_file_pattern=pattern, offload_device="cpu")
            )
        tokenizer_config = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="google/*", offload_device="cpu"
        )

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            use_usp=False,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            local_model_path=wan_dir,
            checkpoint_path=None,
            skip_download=True,
            redirect_common_files=False,
        )

        if use_finetuned:
            print("[IDV2V] Fast-loading finetuned checkpoint (skipping base DiT + VACE-14B)...")
            idv2v_pipeline._load_finetuned_dit_vace(pipe, ckpt, torch_dtype=torch.bfloat16)
        else:
            vace_glob = os.path.join(wan_dir, "Wan-AI/Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors")
            if pipe.vace is None:
                print(f"[IDV2V] Loading VACE selectively from {vace_glob}")
                pipe.vace = load_vace_from_checkpoint(vace_glob, torch_dtype=torch.bfloat16, device="cpu")
            print("[IDV2V] No finetuned checkpoint: using original pretrained weights")

        assert pipe.dit is not None, "DiT was not loaded."
        assert pipe.dit.has_image_input, "Expected an I2V DiT (has_image_input=True)."

        pipe.enable_vram_management(vram_buffer=vram_buffer)

        _PIPE_CACHE["key"] = cache_key
        _PIPE_CACHE["pipe"] = pipe
        return (pipe,)
