# ComfyUI-IDV2V — How To Use

Step-by-step usage guide. For installation and background, see [README.md](README.md). This guide assumes:

- ComfyUI with **ComfyUI-VideoHelperSuite (VHS)** installed (for `Load Video` / `Video Combine`)
- The ID-V2V repo cloned inside this pack's folder and checkpoints downloaded (`bash scripts/download_checkpoints.sh` after `hf auth login`)
- A CUDA GPU. This is a 14B DiT + VACE model: expect single-GPU generation to be slow (upstream tests on 8x A100-80GB; the single-GPU CPU-offload path works but takes time)

---

## 1. The mental model

ID-V2V takes four inputs and produces one output:

| Input | Where it comes from in ComfyUI | Requirements |
|---|---|---|
| Source video | VHS `Load Video` → IMAGE batch | Any length. Drives identity + performance. |
| Stylized first frame | `Load Image` | **Any resolution** (center-cropped internally). Made in any image editor / editing model (e.g. NanoBanana). Does NOT need to perfectly match the source ("imperfect keyframe" is supported — frame 0 follows it, frame 1+ re-align to the source). |
| Text prompt | Sampler's `prompt` field | Required. Describe the stylized scene (upstream's `prompt.txt`). |
| Optional keyframes | `ID-V2V Keyframe` node(s) | Any resolution; aspect ratio within 0.1 of the condition. Index ≥ 1, 0-based. |

The **condition** (what the model is allowed to see of the source) decides the use case:

- **Restylization** (change scene/background/style, keep the person): condition = foreground-on-gray → use SAM3 Segmenter + Foreground-on-Gray.
- **Relighting** (change only lighting, keep everything): condition = the raw source frames → skip segmentation entirely.

---

## 2. Recipe A — Restylization (the default flow)

Node graph:

```
VHS Load Video ──┬─> [ID-V2V Aspect Resize]* ──┬─> ID-V2V SAM3 Video Segmenter ── masks ─┐
                 │                             │                                          v
                 │                             └────────────────────────> ID-V2V Foreground-on-Gray
                 │                                                                        │ condition
Load Image (stylized first frame) ────────────────────────────────────────────┐           v
ID-V2V Pipeline Loader ── pipe ──────────────────────────────────────────────>├──> ID-V2V Sampler ──> VHS Video Combine
                                                                              ┘
* Aspect Resize only needed for 2K/4K sources — see section 6.
```

Steps:

1. **VHS Load Video** — load `source.mp4`. Leave `frame_load_cap = 0` for the full video, or set it to limit frames while iterating.
2. **ID-V2V SAM3 Video Segmenter** — set `sam_prompt` to what to segment (`person` default; `head`, `dog`, ... work). Leave `model_path` empty to use `<repo>/checkpoints/sam3`. Keep `unload_after = true` so SAM3 frees VRAM before the 14B pipeline loads. Connect the `overlay` output to a (muted or collapsed) VHS `Video Combine` to sanity-check the segmentation.
3. **ID-V2V Foreground-on-Gray** — connect the source IMAGE and the masks. Defaults (`gray_value=127`) match upstream.
4. **ID-V2V Pipeline Loader** — leave both paths empty to use `<repo>/checkpoints/idv2v.pth` and `<repo>/checkpoints/wan`. First run loads T5/VAE/CLIP + the finetuned checkpoint; subsequent queue runs reuse the cached pipeline.
5. **ID-V2V Sampler** — connect `pipe`, `condition_0`, `first_frame`. Fill in the `prompt`. Defaults reproduce `scripts/infer.sh`: 1280x720, 81 frames/clip, 30 steps, cfg 5.0, vace_scale 1.0, `ref_pad_num = -1` (SVI anti-drift), seed 123, `max_num_frames = 240`.
6. **VHS Video Combine** — set `frame_rate` to the source fps (visible in VHS Load Video's `video_info`) to match upstream's `OUTPUT_FPS=source` behavior.

An importable version of this graph is in `example_workflows/idv2v_restylization.json`.

First iteration tip: set `steps = 2` on the sampler as a smoke test (upstream's `DEBUG=true`) to validate the whole graph before paying for a full 30-step run.

## 3. Recipe B — Relighting

Change only the lighting; keep the full scene including background. **No segmentation at all**:

```
VHS Load Video ──────────────────────────> Sampler.condition_0   (raw source IS the condition)
Load Image (relit first frame) ──────────> Sampler.first_frame
Pipeline Loader ─────────────────────────> Sampler.pipe
```

The relit first frame must keep everything (including the background) — relight it with an image-editing model, don't restyle it.

## 4. Recipe C — Longer videos

Nothing structural changes. The sampler auto-schedules overlapping clips from the condition length:

- Raise `max_num_frames` to your target length (or `0` for "whole condition").
- Clips overlap by 1 frame (stride = `num_frames_per_clip − 1`); the last clip is end-anchored, so it may overlap more. Each clip is anchored on the previous clip's frame at the splice point, and on overlaps the later clip's frames win — drift-controlled, seamless joins.
- Upstream recommends **more keyframes for longer videos** (Recipe D) to hold the look.
- For SAM3 memory on long sources, set `chunk_size` (e.g. 150–300) on the segmenter — see section 6.

Example: a 240-frame source at defaults → 3 clips of 81 frames, exactly like `scripts/examples/longer_video.sh`.

## 5. Recipe D — Multiple keyframes (e.g. first + last frame)

Pin additional stylized frames at chosen indices:

```
Load Image (kf at 80) ─> ID-V2V Keyframe (frame_index=80) ─┐
Load Image (kf at 160) ─> ID-V2V Keyframe (frame_index=160, keyframes ← chain) ─> Sampler.keyframes
```

Rules (enforced, same as upstream): indices are 0-based and ≥ 1 (index 0 is the first-frame input); no duplicates; index must be < effective output length (`min(condition length, max_num_frames)`); aspect ratio within 0.1 of the condition. Keyframes are injected into the condition and hard-pinned via the VACE mask, so the output lands exactly on them. To pin the last frame of an 81-frame clip, use `frame_index = 80`.

---

## 6. Long videos and 2K/4K sources

Generation always happens at the sampler's `width x height` (720p default — the trained resolution), so native 2K/4K buys nothing downstream and only costs memory. In order of preference:

1. **`ID-V2V Aspect Resize` right after VHS Load Video** (`mode=long_edge`, `size=1280`). The whole graph then runs at ~720p. Right default for 2K/4K.
2. **Segmenter `processing_max_edge` (e.g. 1280)** — keeps the graph at native resolution but runs SAM3 internally downscaled; union masks are nearest-upscaled back. Use when you want a native-res flip test against the source.
3. **Segmenter `chunk_size` (150–300)** — bounds SAM3's per-session memory on long videos. Chunks are independent; ID-V2V uses only the union mask, so cross-chunk ID consistency is irrelevant. If masks flicker at a chunk boundary, raise `chunk_size`.
4. **Segmenter `video_storage_device = cpu`** — session frames in system RAM instead of VRAM.
5. Remember the upstream IMAGE batch itself: a float32 IMAGE is `frames × H × W × 12` bytes (≈100 MB/frame at 4K, ≈11 MB/frame at 720p). Resize early, or process the source in segments via VHS `frame_load_cap` + `skip_first_frames`.

## 7. Using an external chunked SAM3 node

Any node that outputs a standard per-frame MASK batch can replace the built-in segmenter:

```
External SAM3 node ─> ID-V2V Secret Panda Mask Cleanup ─> ID-V2V Foreground-on-Gray ─> Sampler.condition_0
```

- The Cleanup node applies ID-V2V's exact mask post-processing (hole-fill → morph close → bridge gaps → hole-fill).
- Masks at a different resolution than the video: enable `resize_masks_if_needed` on Foreground-on-Gray.
- Multiple masks per frame (per-object): union them first with a mask-combine node — ID-V2V wants ONE union mask per frame.

## 8. The normal+depth variant

Load `idv2v_with_normal_depth.pth` in the Pipeline Loader and feed three conditions: `condition_0` = foreground-on-gray, `condition_1` = surface normals (DAViD), `condition_2` = depth (DepthAnything-V2). Produce normals/depth with the repo's `scripts/idv2v_with_normal_depth/preprocess_with_depth.sh`, or a DepthAnything-V2 ComfyUI node for depth. All three conditions must have the same frame count. **Warning (from upstream):** the two checkpoints share an architecture — loading the wrong one does not error, it silently produces poor output. Match checkpoint to conditions.

---

## 9. Parameter reference (Sampler)

| Parameter | Default | Upstream equivalent | Notes |
|---|---|---|---|
| `width`, `height` | 1280, 720 | `WIDTH/HEIGHT` | Trained at 720p; conditions/first frame/keyframes are center-cropped+resized to this. |
| `num_frames_per_clip` | 81 | `NUM_FRAMES_PER_CLIP` | Must be `4k+1` (45, 49, 81, 121). Enforced. |
| `max_num_frames` | 240 | `MAX_NUM_FRAMES` | Cap on total output frames; `0` = no cap. |
| `steps` | 30 | `NUM_INFERENCE_STEPS` | `2` = smoke test. |
| `cfg_scale` | 5.0 | `CFG_SCALE` | |
| `vace_scale` | 1.0 | `VACE_SCALE` | Strength of the condition. |
| `ref_pad_num` | -1 | `REF_PAD_NUM` | `-1` = full anti-drift (SVI) padding with the first frame; `0` = original VACE zero padding; `N` = N-frame padding. |
| `seed` / `different_seed_per_clip` | 123 / off | `SEED` / `--different_seed_per_clip` | Per-clip seed = `seed + 42*clip_idx` when on. |
| `negative_prompt` | empty | `NEGATIVE_PROMPT` | Empty → Wan 2.1's default Chinese quality-degradation prompt. |

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Pack fails to run with "could not find the ID-V2V repository" | Clone the repo inside `ComfyUI-IDV2V/` or set `IDV2V_REPO`. |
| "a different `diffsynth` package shadows the ID-V2V fork" | `pip uninstall diffsynth` — the pipeline needs the repo's bundled fork. |
| SAM3 download / auth errors | SAM3 is gated: `hf auth login` with a token that has read access to `facebook/sam3`, then re-run `download_checkpoints.sh`. |
| `transformers` import errors from another node pack | SAM3 needs `transformers>=5.6`; another pack may pin 4.x. Use a dedicated venv. |
| OOM during SAM3 | Set `chunk_size` (150–300), `processing_max_edge=1280`, or `video_storage_device=cpu`; or Aspect-Resize before the segmenter. |
| OOM when the Wan pipeline loads | Ensure the segmenter's `unload_after=true`; lower the loader's `vram_buffer`; close other GPU consumers. |
| "num_frames_per_clip must be 1 plus a multiple of 4" | Use 45/49/81/121. |
| "Keyframe index N >= effective output length" | The keyframe lies beyond `min(condition length, max_num_frames)` — raise `max_num_frames` or lower the index. |
| Output looks structurally wrong / mushy with no error | Wrong checkpoint for the condition count (default vs normal+depth variant) — they load interchangeably but are not interchangeable. |
| Masks flicker between segments | Chunk-boundary detection differences — raise `chunk_size`. |
| Output plays at wrong speed | Set VHS Video Combine's `frame_rate` to the source fps (shown in VHS Load Video `video_info`). |
| A benign `diffsynth` warning about `--use_multi_control_vace` with 1 condition | Expected upstream behavior; safe to ignore. |
