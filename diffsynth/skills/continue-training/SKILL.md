---
name: continue-training
description: Continue/resume a Wan2.2-S2V LoRA training run from its latest (or a chosen) checkpoint. Generates a new `..._leo_<N+1>.sh` SLURM script that resumes via --lora_checkpoint with a configurable --skip_frames, following the project's `_cont<STEP>_SF<SF>` convention. Can also queue the continuation behind a still-running job (--depends-on) and resolve the newest checkpoint at launch time (--runtime-latest). Use when the user says "continue training X", "resume the run with skip frames N", "pick up from the latest checkpoint", or "queue a continuation behind the current job".
---

# Continue a training run

Generate a continuation SLURM script that resumes an existing run from a checkpoint,
following the project convention used across `live_dealer/train/*_leo_<N>.sh`.

## What a continuation script changes (vs. the base script)

Everything is preserved except the launch args:
- `--output_path` gets a `_cont<TOTAL>_SF<SF>` suffix (a **new** run dir; the source run's
  checkpoints are never touched). A pre-existing `_cont…_SF…` suffix is stripped first so
  chained continuations don't grow unbounded.
- `--lora_checkpoint <run>/step-<STEP>.safetensors` is added/replaced.
- `--skip_frames <SF>` is added/replaced.
- `#SBATCH --job-name` becomes a concise `<run>_sf<SF>_cont<TOTAL>`.

`<STEP>` defaults to the **latest** `step-*.safetensors` in the run's output dir — this is
the actual checkpoint file the run resumes from.

`<TOTAL>` is the **cumulative step count from the very first run**, not the last job's
local step. A continuation run restarts its own step counter at 0, so `<TOTAL>` =
the `_cont<N>` already encoded in the parent run name (0 if this is the first
continuation) + the resume `<STEP>`. This keeps run-dir names monotonic across a chain
of continuations (e.g. `_cont600_SF3` → resume its latest `step-400` → `_cont1000_SF3`).

## How to run it

Use the helper (don't hand-edit unless the base script is unusual):

```bash
python3 .claude/skills/continue-training/continue_training.py \
  --base <path/to/base_leo_N.sh> \
  --skip-frames <N> \
  [--step <STEP>]        # default: latest checkpoint on disk \
  [--runtime-latest]     # resolve newest checkpoint AT LAUNCH (see below) \
  [--depends-on <JOBID>] # queue behind a job: sbatch --dependency=afterany:<id> \
  [--out <path>]         # default: next *_leo_<N+1>.sh \
  [--submit]             # default: write only, print the sbatch command
```

Default behavior writes the script and prints the suggested `sbatch` command **without
submitting** — let the user review first (they often want to resume from a newer
checkpoint than the one currently on disk). Only pass `--submit` when the user has
clearly asked to launch it now.

## Queuing behind a still-running run (`--runtime-latest` + `--depends-on`)

When the source run is **still training** (or might die near the end) and you want to
line up the continuation now, don't hardcode a step — the checkpoint doesn't exist yet,
and if the run crashes before reaching it a hardcoded path would fail on launch.

- `--runtime-latest` emits a script that, **when the SLURM job actually starts**, globs
  the newest `step-*.safetensors` on disk, derives a truthful `_cont<STEP>_SF<SF>` output
  dir from that real step (plus any steps the parent already accumulated), and feeds both
  into the launch via `RESUME_CKPT` / `RESUME_RUN_OUT` container env vars. No checkpoint
  needs to exist at generation time. It exits 1 with a clear message if none exists at
  launch. (Targets the project's container template: `srun` + `singularity exec` +
  `"$SIF"`; fails clearly on other layouts.)
- `--depends-on <JOBID>` submits with `--dependency=afterany:<JOBID>` — **afterany**, not
  afterok, so the continuation still runs (and grabs the latest checkpoint) even if the
  source job fails late. Because the dependent job starts only after the source ends, all
  its checkpoints exist by then.

Example — queue an SF100 continuation behind running job 12345:
```bash
python3 .../continue_training.py --base <base>.sh --skip-frames 100 \
  --runtime-latest --depends-on 12345 --out <base>_cont_SF100.sh --submit
```
In this mode the printed run name shows `_cont<latest+<parent>>_SF<SF>` since the exact
step is only known at launch; the `#SBATCH --job-name` uses `cont latest`.

## Procedure for the agent

1. Identify the base script. If the user names a run rather than a file, find the
   `*_leo_<N>.sh` whose `--output_path` matches; the most recently opened/edited script
   is a strong default.
2. Run the helper (without `--submit`) to generate the script. It prints the available
   checkpoints and which one it picked.
3. Show the user the generated path, the resume checkpoint, the new run name, and the
   `sbatch` command. Confirm before submitting.
4. On confirmation, `sbatch` it (or re-run with `--submit`) and report the job id +
   `squeue` status.

## Checkpoint-path resolution

The container sees `--output_path /output/<run>`; the host dir is
`$SCRATCH_PROJECT_DIR/outputs/<run>` (the `--bind ${OUTPUT_DIR}:/output` mount). The
helper derives this from the base script automatically; override with `--outputs-dir`
if a script uses a non-standard layout.

## Notes / gotchas

- Submitting an N-node job is outward-facing and hard to reverse — confirm the
  checkpoint step with the user, since an active run may write a newer one any minute.
- The helper requires the base name to match `*_leo_<N>.sh`; pass `--out` for other names.
- It anchors the inserted args after the `--lora_rank` line. If a base script has no
  `--lora_rank`, the helper errors — handle that case manually.
