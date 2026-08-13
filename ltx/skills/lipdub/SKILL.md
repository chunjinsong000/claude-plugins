---
name: lipdub
description: Set up and run LTX-2.3 video dubbing (LipDub) — re-dub the speech in an existing video to new dialogue while preserving the speaker's appearance, scene, and motion, using the IC-LoRA on top of the LTX-2.3-22B distilled model. Covers cloning the LTX-2 code, the Hugging Face gate/auth requirement, downloading the gated weights (base checkpoint + spatial upscaler + LipDub IC-LoRA + Gemma text encoder), and the exact lipdub CLI. Use when the user says "run LTX lipdub", "dub this video with LTX", "set up LTX-2 lipdub / lipsync", "download the LTX dubbing model", or references ltx.io lipsync-vs-lipdub.
---

# LTX-2.3 LipDub — video dubbing

Re-dub the spoken dialogue in an existing video to a new script/language while
keeping the speaker's face, camera, body language and scene intact. LipDub is a
**distilled-checkpoint, single-IC-LoRA, two-stage** pipeline in the
[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) repo
(`packages/ltx-pipelines/src/ltx_pipelines/lipdub.py`).

Reference: [ltx.io/blog/lipsync-vs-lipdub](https://ltx.io/blog/lipsync-vs-lipdub),
[docs.ltx.video LipDub guide](https://docs.ltx.video/open-source-model/usage-guides/lip-dub-beta).

Lipsync vs LipDub: **lipsync** *generates* a talking video from audio (no source
footage); **LipDub** *transforms* existing footage — you give it the video + a
prompt with the new dialogue and it rewrites only the mouth.

## Layout

Code lives at `LTX2_ROOT` (default `/home/ubuntu/chunjin/project/valka-ai/LTX-2`).
Weights go under `$LTX2_ROOT/models/`. The two helper scripts here drive the
download and the run; override the location with `export LTX2_ROOT=/path/to/LTX-2`.

## Step 1 — Code (one time)

```bash
cd /home/ubuntu/chunjin/project/valka-ai
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Lightricks/LTX-2.git   # if not already present
```
(Weights are pulled via the `hf` CLI below, so `git-lfs` is not required for the code.)

## Step 2 — Hugging Face gate + auth (USER action — cannot be automated)

All three model repos are **gated**. The user must, with their own HF account:

1. Accept the license on each page:
   - https://huggingface.co/Lightricks/LTX-2.3 (base checkpoint + spatial upscaler)
   - https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub (the LipDub LoRA)
   - https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized (Gemma text encoder)
2. Authenticate the machine: `hf auth login` (paste a **Read** token; fine-grained
   tokens need the "read gated repos" scope) — or `export HF_TOKEN=hf_xxx`.
   Do NOT ask the user to paste the token into chat; have them run `hf auth login`.

Check with `hf auth whoami`. A 401/403 during download means the gate wasn't
accepted or the token lacks gated-repo read scope.

## Step 3 — Download weights (~40–60 GB)

```bash
bash download_lipdub_models.sh
```
Pulls into `$LTX2_ROOT/models/`:
- `ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors` — distilled base (LipDub uses the distilled model)
- `ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` — required by the two-stage pipeline
- `ltx-2.3/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` — the LipDub IC-LoRA
- `gemma-3-12b/` — Gemma text encoder (all files)

## Step 4 — Run

```bash
bash run_lipdub.sh <reference_video.mp4> "<new dialogue prompt>" [output.mp4]
```
which runs (via `uv run`, auto-installing deps from `uv.lock` on first invocation):

```bash
uv run python -m ltx_pipelines.lipdub \
    --distilled-checkpoint-path models/ltx-2.3/ltx-2.3-22b-distilled-1.1.safetensors \
    --spatial-upsampler-path    models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --gemma-root                models/gemma-3-12b \
    --lora                      models/ltx-2.3/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors \
    --reference-video           <video> \
    --reference-strength        1.0 \
    --prompt                    "<new dialogue>" \
    --seed 42 --output-path <output.mp4>
```

### CLI facts (verified against the source)

- Entry point: `python -m ltx_pipelines.lipdub`.
- **Required**: `--distilled-checkpoint-path`, `--spatial-upsampler-path`,
  `--gemma-root`, exactly one `--lora`, `--reference-video`.
- **No `--num-frames` / `--frame-rate`** — frame count and fps are read from the
  reference video (frame count is silently snapped down to the nearest `8k+1`).
- `--prompt` is the NEW dialogue/description the mouth is dubbed to.
- `--reference-strength` (default 1.0) controls IC-LoRA video-reference conditioning.
- `--height` / `--width` default to the model's stage-2 size; must be divisible by 64.

### Low VRAM (the base is 22B)

Append to the run command: `--quantization fp8-cast --offload cpu`
(`--offload disk` uses even less host RAM but is slower; `--compile` speeds up
repeat runs). Choices come from `QUANTIZATION_POLICIES` / `OffloadMode` in
`packages/ltx-pipelines/src/ltx_pipelines/utils/args.py`.

## Notes

- ComfyUI alternative: copy the LoRA into `models/loras` and use the official
  LipDub workflow from https://github.com/Lightricks/ComfyUI-LTXVideo/.
- Related lightweight project: JustDubIt.
