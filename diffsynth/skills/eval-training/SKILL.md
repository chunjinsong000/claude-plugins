---
name: eval-training
description: Generate the evaluation command for a Wan2.2-S2V training run (e.g. leo_114.sh, a SLURM job id, or a checkpoint path like outputs/<run>/step-600.safetensors). Reads the training script and emits the inference snippet on the eyes-only test set, mirroring the canonical evals in live_dealer/infer/infer_card_class.sh — correct WAN_* env flags, lora_path/step, width/height, and pose/object inputs. Emits either the default bidirectional livedealer_infer.py command or, with --streaming, the real-streaming (4-step causal, full-length video) livedealer_infer_real_streaming.py torchrun command. Use when the user says "evaluate this training job", "make the infer command for run X", "eval job 43983676", "eval this checkpoint", "generate the eval snippet", or "eval in streaming mode / full 10s".
---

# Generate an evaluation snippet for a training run

Produce the `livedealer_infer.py` command that evaluates a training run on the
eyes-only test set, following the pattern at the bottom of
`live_dealer/infer/infer_card_class.sh`.

## What it derives from the training script

| inference field | source in the training `*_leo_<N>.sh` |
|---|---|
| `WAN_*` env flags | tokens preceding `accelerate launch` (e.g. `WAN_CARD_SEQUENTIAL=true`) |
| `--lora_path outputs/<run>/step-<STEP>` | `<run>` = basename of `--output_path`; STEP = latest checkpoint (or `--step`) |
| `--width` / `--height` | training `--width` / `--height` |
| `--pose_video` / `--object_video` | included only if `s2v_pose_video` / `s2v_object_video` is in `--extra_inputs` |
| `--save_path output/<run>-<STEP>` | derived |

Fixed (eyes-only test set): `--input_image`, `--audio_path`, `--gt_path`, the pose/object
list files, and constants `--infer_frames 12 --num_clips 1 --no_motion_video --use_block_attn`,
prefixed with `HF_HUB_OFFLINE=1`.

## How to run it

```bash
python3 .claude/skills/eval-training/eval_snippet.py \
  ( --base <path/to/training_leo_N.sh> | --job-id <SLURM_JOB_ID> | --checkpoint <outputs/<run>/step-N.safetensors> ) \
  [--step <STEP>]   # default: latest checkpoint (ignored with --checkpoint) \
  [--csv [metadata.csv]]   # evaluate on a metadata CSV (bare --csv auto-picks) instead of the lists \
  [--distill[=T0,T1,...]]  # distillation model: denoise in a few fixed steps \
  [--streaming]     # emit the real-streaming (4-step causal) eval, full video (see below) \
  [--append]        # also append the snippet to infer_card_class.sh \
  [--launch]        # also run the eval on a GPU node (srun + singularity)
```

## CSV test set (`--csv`)

By default the snippet evaluates on the hard-coded eyes-only list files. Pass
`--csv` to evaluate on a metadata CSV instead — the helper emits
`--dataset_metadata_path <csv> --dataset_base_path <DIR>` (both natively supported
by `livedealer_infer.py`) in place of the per-input `--pose_video` /
`--object_video` / `--input_image` / `--audio_path` / `--gt_path` /
`--card_detection` list flags.

- **Bare `--csv` auto-picks** the canonical 100-sample eyes-only CSV from the config:
  - card-detection models (`card_detection` in `--extra_inputs`, e.g.
    `WAN_CARD_CLASS_EMBED`) → `..._leo_cards_sample100.csv` (has the `card_detection`
    column);
  - card-encoder models (use `s2v_object_video`, no card detection) →
    `..._sample100_no_carddet.csv`.
  - Pass an explicit `--csv <metadata.csv>` to force a specific file.

- Expected CSV columns: `video, input_audio, s2v_pose_video, s2v_object_video,
  input_image, card_detection`. The CSV carries **every** per-input list, so the
  `--extra_inputs` pose/object/card gating is bypassed — the pipeline uses whichever
  inputs its WAN_* config needs.
- `--csv` is **not** a run identifier; combine it with `--base`/`--job-id`/`--checkpoint`
  for the training config (WAN_* flags, resolution).
- An **absolute** CSV path under the `data/` symlink is rewritten workspace-relative
  (e.g. `/leonardo_work/.../shared/livedealer/test_set/foo.csv` →
  `data/project21_snapshot_12032025_packed/test_set/foo.csv`). `--dataset-base-path`
  defaults to `data/project21_snapshot_12032025_packed`; override if the CSV's
  relative paths resolve against a different root.
- The save path / log gets a tag derived from the CSV filename (e.g. `-sample100`)
  so a CSV eval doesn't clobber a list-file eval of the same checkpoint. Override
  with `--save-tag`.

## Three ways to identify the run

Provide **one** of:

- `--base <training_leo_N.sh>` — read the script directly; run name = basename of
  `--output_path`, step = latest checkpoint (or `--step`).
- `--job-id <SLURM_JOB_ID>` — resolve the training script from the job (see below).
- `--checkpoint <PATH>` — point straight at a `step-<N>.safetensors`. The **run name
  and step are derived from the path**, so the checkpoint dir need not match any
  training script's `--output_path` (the common case for checkpoints copied in from
  another machine, e.g. `outputs/b200_card_class_embed_121/step-600.safetensors`).

`--checkpoint` still needs the training **config** (WAN_* flags, resolution,
extra_inputs gating). Supply it either way:

- **Combine with `--base`/`--job-id`** — pulls the config from that script but takes
  the run/step from the checkpoint path. Best when a matching training script exists
  but its `--output_path` differs from the checkpoint dir.
- **Set it explicitly** — `--wan-env WAN_CARD_CLASS_EMBED=true` (repeatable, or one
  comma/space-separated string), `--extra-inputs input_image,input_audio,s2v_pose_video,card_detection`,
  `--width`, `--height`. With no source for a field the helper warns and defaults
  (no WAN_* flags; `--extra_inputs` = `input_image,input_audio,s2v_pose_video,s2v_object_video`).

## Streaming mode (`--streaming`)

By default the skill emits the **bidirectional** `livedealer_infer.py` command
(25-step, one short clip per sample). Pass `--streaming` to instead emit the
**real-streaming** eval — a `torchrun` launch of
`live_dealer/infer/livedealer_infer_real_streaming.py` (4-step causal generation
with a KV cache) that produces the **full-length video** (`num_clips` auto-caps to
the pose/audio length at runtime, e.g. 100 → 25 clips for a 10 s / 300-frame clip).

Key differences the helper handles:

- **Per-sample loop, not one command.** The streaming launcher reads
  pose/card/audio/gt as `.txt` lists but takes only a **single** `--input_image`
  per run (it builds the ref cache once). To keep each clip's dealer matched, the
  snippet is a `paste … | while read` loop over the index-aligned test lists, one
  `torchrun` invocation per clip. (Each invocation reloads + FP8-quantizes the DiT,
  so it's heavier than the single-process list eval — that's the price of a matched
  ref image per sample.)
- **Cards via `--card_detection` only.** Streaming dropped `card_video`; only
  `WAN_CARD_CLASS_EMBED` (raw-detection) models are card-conditioned. The launcher's
  CLI auto-sets `WAN_CARD_CLASS_EMBED` from `--card_detection`. A card-encoder
  (`s2v_object_video`) model gets **no** card signal in streaming — the helper warns.
- **Incompatible with `--csv`** (the streaming launcher has no
  `--dataset_metadata_path`); the helper errors if both are given.
- **`--distill` is ignored** — streaming already denoises in `--num_inference_steps`
  (default 4) fixed steps.
- Output goes to `output/<run>-<step>-stream/` (`-stream` suffix so it never
  clobbers a bidirectional eval of the same checkpoint).

Streaming knobs (all optional, sensible defaults): `--gpus` (torchrun
`--nproc_per_node`, default `gpu` = all visible; pass `1` for single-GPU),
`--num-inference-steps` (4), `--frames-per-clip` (12), `--lframes-per-block` (3),
`--num-clips` (100 → full video), `--motion-frames` (9), `--fps` (30),
`--rope-offset` (`relative`), `--save-layout` (`left_gt_right_gen`). Width/height
come from the training config as usual (default 1280×720 if the config has none).

> **Architecture caveat.** Streaming is built for **causal / few-step distilled**
> checkpoints. Running a **bidirectional 25-step** LoRA (e.g. a plain
> `card_class_embed` run) through the 4-step causal pipeline is plumbing-valid but
> quality is not guaranteed — validate one sample before a full sweep.

> **`--launch` note.** The `--launch` srun requests `--gres=gpu:1`, so a launched
> streaming eval is **single-GPU** (`--gpus 1`). Multi-GPU streaming (the 4-GPU
> `livedealer_infer_real_streaming.sh` layout) also needs that script's module loads
> and NCCL env, so for multi-GPU run it from that wrapper rather than `--launch`.

## Distillation models (`--distill`)

For a distillation-trained model, the eval should denoise in the model's few fixed
timesteps rather than the default ~25-step schedule. Pass `--distill` to emit
`--custom_timesteps 1000,768,358` (the schedule from `direct_distill_loss()` in
`diffsynth/pipelines/wan_video_new.py`). Override the steps with
`--distill=1000,500,250`. In distill mode the helper also adds `--no_tea_cache`
(the many-step skip heuristic is meaningless at 3 steps) and tags the save path /
log with `-distill` so it doesn't clobber a regular eval of the same checkpoint.

`livedealer_infer.py` accepts `--custom_timesteps T0,T1,...` directly and forwards
it to the pipeline's `custom_timesteps` (see `__call__`); `--distill` is just the
convenience wrapper that fills in the canonical schedule.

### Frozen extra modules (`--extra_module_ckpt_path`) — avoids all-black video

A `--task direct_distill` run freezes non-LoRA modules like `card_encoder` and does
**not** save them in its checkpoint (the checkpoint is LoRA-only). `card_encoder` is
created at inference via `.to_empty()` (uninitialized memory); if nothing fills it,
`model_fn` runs the object/card latents through garbage weights → NaN → **all-black**
output (independent of the denoising schedule). To fix this, the helper auto-emits
`--extra_module_ckpt_path <base>` whenever the training script has `--task
direct_distill` (or you pass `--distill`), pointing at the base checkpoint the run
resumed from (its `--lora_checkpoint`, converted from the `/output/...` container path
to the host-relative `outputs/...`). `livedealer_infer.py` then loads only the
**non-LoRA** tensors (e.g. `card_encoder.weight/bias`) from that base, leaving the
distill LoRA (which is cumulative — it continued the base LoRA) untouched.

If a distill run resumed from a base that itself lacks the extra module, or the
auto-derived path is wrong, override or drop `--extra_module_ckpt_path` in the
printed snippet.

Pass **either** `--base` (the training script directly) **or** `--job-id` (a SLURM
job id). With `--job-id` the helper resolves the training script automatically:
1. `scontrol show job <id>` → `Command=` (while the job is running / still in the
   controller's memory);
2. falling back to the job's stdout log
   `/leonardo_scratch/large/userexternal/$USER/slurm_logs/*_<id>.out`, which echoes
   `The original script is located at: <script>` (for finished jobs whose logs survive).
Override the log location with `--slurm-logs-dir '<template with {user}>'` if needed.
The resolved `job <id> -> <script>` mapping is printed to stderr.

By default it prints the snippet to stdout (checkpoint list goes to stderr). Show the
snippet to the user; only pass `--append` when they want it added to
`live_dealer/infer/infer_card_class.sh`.

## Auto-launch (`--launch`)

When the user asks to **run / launch / execute** the eval (not just generate it),
pass `--launch`. The helper wraps the generated snippet in a non-interactive
`srun` + `singularity exec --nv` invocation — the same launch wrapper documented
at the top of `infer_card_class.sh`, but turned into a single batchable command
(no `--pty bash` shell). It:

- requests `1×GPU` on `boost_qos_dbg` for `00:30:00` (override with `--qos`,
  `--time`, `--account`, `--partition`, `--cpus`, `--mem`),
- binds `/workspace`, `/workspace/models` (`wan_models`), and `/workspace/outputs`
  the same way the interactive wrapper does (override the repo root with
  `--workspace`, default cwd; image with `--sif`, default `$WORK/<SIF_REL>`),
- tees all output to `eval_logs/eval_<run>-<step>.log` under the workspace, and
- blocks until the srun finishes, exiting with the srun's return code.

Because it blocks (model load + inference can take minutes), **the agent should
invoke the helper with `--launch` as a background Bash command** (`run_in_background:
true`) and report the Slurm job id + log path, then surface the result on
completion. Example:

```bash
python3 .claude/skills/eval-training/eval_snippet.py --job-id 44063494 --launch
```

## Procedure for the agent

1. Identify the training run. If the user gives a SLURM job id (e.g. "eval job
   43983676"), pass `--job-id`. If they name/point at a checkpoint file (e.g. "eval
   this step-600" or `outputs/<run>/step-600.safetensors`), pass `--checkpoint` —
   and if a matching training script exists, also pass `--base` for its config (run
   name + step still come from the checkpoint path). Otherwise identify the training
   script (the most recently opened `*_leo_<N>.sh` is a strong default; or match the
   run name to a script's `--output_path`) and pass `--base`.
2. Run the helper and present the generated snippet (and which step it used).
3. If the user wants it saved, re-run with `--append` or add it where they prefer.
4. If the user wants it **run**, re-run with `--launch` (in the background) and
   report the job id + `eval_logs/` path; verify the checkpoint and test-set list
   files exist first (the helper fails fast if the checkpoint dir is missing).

## Notes / gotchas

- WAN_* flags are mirrored verbatim from training to inference — that's the convention in
  infer_card_class.sh (e.g. `WAN_CARD_COMPRESSION`, `WAN_CARD_ENCODER`, `WAN_CARD_SEQUENTIAL`).
- The eyes-only list files are hard-coded to match recent evals. If a run needs a
  different test set, edit the printed snippet.
- Checkpoint paths use `outputs/<run>/...` (host/workspace-relative), as the inference
  runs from /workspace; this differs from the `/output/<run>` container path in training.
