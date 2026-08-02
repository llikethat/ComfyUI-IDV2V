# ComfyUI-IDV2V

ComfyUI custom nodes for [Eyeline-Labs/ID-V2V](https://github.com/Eyeline-Labs/ID-V2V) — identity-preserving video restylization and relighting (Wan2.1-I2V-14B + VACE, SIGGRAPH Asia 2026).

**New here? See [USAGE.md](USAGE.md)** for step-by-step recipes (restylization, relighting, longer videos, keyframes), the sampler parameter reference, and troubleshooting.

Video/image loading and saving is deliberately left to core ComfyUI nodes and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) (VHS `Load Video` / `Video Combine`). This pack only provides the ID-V2V-specific stages.

## Nodes

| Node | Wraps | Purpose |
|---|---|---|
| **ID-V2V Aspect Resize (2K/4K)** | — | Aspect-preserving batch rescale (long/short edge or W/H target, divisibility rounding, mini-batched). Put it right after the video loader for 2K/4K sources. |
| **ID-V2V SAM3 Video Segmenter (chunked)** | `idv2v.preprocess.sam3` | Text-prompted SAM3 video segmentation with tracking + Secret Panda mask cleanup. Supports temporal chunking, a downscaled processing resolution, and CPU frame storage for long/high-res videos. Outputs per-frame union MASK + debug overlay IMAGE. |
| **ID-V2V Secret Panda Mask Cleanup** | `idv2v.preprocess.secret_panda` | Standalone MASK→MASK cleanup (hole-fill → close → bridge → hole-fill). Use it to bring masks from *external* segmentation nodes (e.g. a chunked SAM3 node from another pack) up to ID-V2V spec. |
| **ID-V2V Foreground-on-Gray Condition** | `idv2v.preprocess.orig_pixel` | Keeps pixels inside the mask, grays out the rest — the VACE control signal for the default `idv2v` model. |
| **ID-V2V Pipeline Loader** | model-loading half of `idv2v/inference/pipeline.py` | Loads Wan2.1 T5 + VAE + CLIP and fast-loads the finetuned `idv2v.pth` DiT+VACE (mmap). Cached across runs. |
| **ID-V2V Keyframe** | `--key_frame_paths/--key_frame_indices` | Pins a stylized frame at index N ≥ 1. Chainable for multiple keyframes. |
| **ID-V2V Sampler (multi-clip)** | generation half of `idv2v/inference/pipeline.py` | Clip-by-clip I2V + VACE generation with keyframe hard-pinning, SVI anti-drift padding, splice-frame chaining, and overlap stitching. Outputs an IMAGE frame batch. |
| **ID-V2V Flip Test** | `idv2v/inference/flip_test.py` | Builds the A/B "Generated ↔ Source" flip visualization as frames (encode with VHS). |

## Installation

1. Clone this pack into `ComfyUI/custom_nodes/`.

2. Clone the ID-V2V repo **inside this folder** (recommended) or anywhere and set `IDV2V_REPO`:

   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-IDV2V
   git clone https://github.com/Eyeline-Labs/ID-V2V
   ```

   The nodes import `idv2v` and the repo's bundled `diffsynth` fork directly from that clone — do **not** `pip install diffsynth` (a stock DiffSynth-Studio lacks the custom `wan_video_new_multiVace_svi` pipeline and will be rejected at load time).

3. Install Python deps into the ComfyUI environment:

   ```bash
   pip install -r requirements.txt
   ```

   Notes:
   - `transformers>=5.6` is required for SAM3 (`Sam3VideoModel`). If another node pack pins an older transformers, use a separate venv.
   - `flash-attn` is strongly recommended for the 14B attention path.
   - The upstream repo targets torch 2.6; newer torch generally works for single-GPU inference but is untested upstream.

4. Download checkpoints (~96 GB) using the repo's own script. SAM3 is gated on Hugging Face, so authenticate first:

   ```bash
   cd ID-V2V
   hf auth login          # token with read access to facebook/sam3
   bash scripts/download_checkpoints.sh
   ```

   This populates `ID-V2V/checkpoints/` with `idv2v.pth`, `sam3/`, and `wan/` (T5 + VAE + tokenizer + CLIP) — the defaults the loader and segmenter nodes look for when their path fields are left empty.

## Workflows (mapping to the repo's recipes)

An importable example is in `example_workflows/idv2v_restylization.json`.

**Restylization / imperfect keyframe** (= `scripts/examples/restylization.sh`):

```
VHS Load Video ─┬─> SAM3 Segmenter ─> Foreground-on-Gray ─> Sampler.condition_0
                └──────────────────────────────────────────> (optional Flip Test.source)
Load Image (stylized first frame) ─────────────────────────> Sampler.first_frame
Pipeline Loader ───────────────────────────────────────────> Sampler.pipe
Sampler.frames ────────────────────────────────────────────> VHS Video Combine
```

**Relighting** (= `scripts/infer_relighting.sh`): skip the SAM3 + Foreground-on-Gray nodes entirely — connect the raw source frames from `VHS Load Video` directly to `Sampler.condition_0`. The first frame should be a relit version of the source's first frame with the background kept.

**Longer video** (= `scripts/examples/longer_video.sh`): nothing extra — the sampler auto-schedules overlapping 81-frame clips from the condition length; raise `max_num_frames`.

**First + last frame / multiple keyframes** (= `scripts/examples/first_last_frame.sh`): add `ID-V2V Keyframe` node(s) (e.g. `frame_index=80` for the last frame of an 81-frame clip) and connect the chain to `Sampler.keyframes`.

**Normal+depth variant** (`idv2v_with_normal_depth.pth`): point the loader at that checkpoint and feed surface-normal and depth condition videos into `condition_1` / `condition_2` (produce them with the repo's `scripts/idv2v_with_normal_depth/preprocess_with_depth.sh`, or any DepthAnything-V2 ComfyUI node for depth). The two checkpoints share an architecture, so loading the wrong one does not error — it silently produces poor output. Match checkpoint to conditions.

## Long videos and 2K/4K sources

Generation always runs at the sampler's `width x height` (720p by default — that is what the model was trained for), so native 2K/4K frames buy nothing downstream of the conditions; they only cost VRAM and RAM. Recommended graph for high-res sources:

```
VHS Load Video (4K) ─> ID-V2V Aspect Resize (long_edge=1280) ─> SAM3 Segmenter ─> Foreground-on-Gray ─> Sampler
```

Levers, in order of preference:

1. **Aspect Resize after the loader** — the whole graph then works at ~720p. This is the right default for 2K/4K.
2. **SAM3 `processing_max_edge`** — if you want to keep the video at native resolution in the graph (e.g. for a 4K flip-test against the source), the segmenter can still run internally at e.g. 1280 long-edge and nearest-upscale the union masks back to input resolution.
3. **SAM3 `chunk_size`** — SAM3 keeps the whole session's frames on the inference device, which OOMs on long videos. Chunking (e.g. 150–300 frames) bounds that. Chunks are independent: the text prompt re-detects the concept per chunk, and since ID-V2V only consumes the **union** mask, cross-chunk object-ID consistency doesn't matter. If detections flicker at a chunk boundary, raise `chunk_size`.
4. **SAM3 `video_storage_device=cpu`** — stores session frames in system RAM instead of VRAM (slower, larger capacity).
5. **Sampler length** — already unbounded: it schedules overlapping 81-frame clips for any condition length (`max_num_frames=0` disables the cap). Conditions are converted frame-by-frame and immediately center-cropped to the target resolution, so long batches never exist as full-resolution PIL lists in RAM. The practical ceiling is your ComfyUI IMAGE batch upstream: a float32 IMAGE tensor is `frames x H x W x 12` bytes (4K ≈ 100 MB/frame, 720p ≈ 11 MB/frame) — another reason to Aspect-Resize early or load in segments.

**Reference images**: the stylized first frame and keyframes can be **any resolution** — the sampler center-crops/resizes them to the target. The only constraint (inherited from upstream) is that keyframe aspect ratio must be within 0.1 of the condition video's.

## Using an external / chunked SAM3 node instead

The segmentation stage is swappable. The only contract `Foreground-on-Gray` needs is a **per-frame MASK batch** (one mask per source frame). To substitute a SAM3 node from another pack:

```
Your SAM3 node ─> ID-V2V Secret Panda Mask Cleanup ─> ID-V2V Foreground-on-Gray ─> Sampler.condition_0
```

- `Secret Panda Mask Cleanup` applies the exact post-processing ID-V2V's own preprocessing uses, so external masks behave identically downstream.
- If the external node runs at a different resolution than the video, enable `resize_masks_if_needed` on `Foreground-on-Gray` (nearest-neighbor; upstream `orig_pixel.py` is strict about matching resolutions, hence off by default).
- If the external node outputs multiple masks per frame (per-object), merge them to a union first (e.g. with a mask-combine node) — ID-V2V consumes one union mask per frame.

## Practical notes

- **Resolution / frames**: generation targets 720p (`1280x720`); `num_frames_per_clip` must be `4k+1` (45, 49, 81, 121…). Conditions, first frame, and keyframes are center-cropped/resized to the target internally, mirroring the repo.
- **VRAM**: this is a 14B DiT + VACE. The loader uses the repo's CPU-offload VRAM management (single-GPU path; the repo's torchrun/USP multi-GPU mode does not apply inside ComfyUI). Expect it to be slow on one card, exactly as upstream documents. Leave `unload_after=True` on the SAM3 node so SAM3 is freed before the Wan pipeline allocates.
- **System RAM**: the finetuned checkpoint is memory-mapped on load; loading is fast from a local SSD.
- **Prompt**: required (the repo reads it from `prompt.txt`; here it's the sampler's prompt field). Empty negative prompt falls back to Wan 2.1's default Chinese quality-degradation prompt.
- **FPS**: the sampler outputs frames only; set the frame rate on your VHS `Video Combine` node (use the source video's fps to match upstream's `OUTPUT_FPS=source` behavior — VHS `Load Video` reports it in `video_info`).

## License

Node code in this pack: MIT. ID-V2V itself and its checkpoints are governed by the upstream repository's license — review it before use.
