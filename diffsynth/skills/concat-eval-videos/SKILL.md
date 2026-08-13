---
name: concat-eval-videos
description: Concatenate per-clip eval output videos into one timeline per run, and optionally build a left/right side-by-side comparison of two or more runs (with a clip index + filename overlay). Reads the per-clip mp4s a livedealer_infer.py eval writes under output/<run>-<step>-.../, orders them by a metadata CSV (or filename), keeps only clips common to all runs so panels stay frame-aligned, and writes the result. Must run inside the singularity container (use the launch wrapper). Use when the user says "stack/concatenate the eval videos", "make a side-by-side comparison of these two runs", "compare method A vs B", or "render the index/filename on the comparison".
---

# Concatenate & compare eval videos

Turn the per-clip mp4s an eval run produces (one file per clip under
`output/<run>-<step>-<set>/`) into:

- **one concatenated video per run** (all clips played in order), and
- **a left/right side-by-side comparison** of two or more runs, frame-aligned per
  clip, with an optional top-left overlay of the clip **index + filename**.

This is the generalization of the one-off `live_dealer/infer/stack_compare_sample100.py`.

## Why it must run in the container

The login node has **no `ffmpeg`** and no `imageio` in its python. The compositor
relies on `imageio` + `imageio-ffmpeg` (bundled binary), which only exist inside the
project singularity image. Always run it via the launch wrapper (`srun` +
`singularity exec --nv`), never directly on a login node.

## How the inputs are matched

| field | source |
|---|---|
| clip files | `*.mp4` in each `--runs` dir (named by the clip's GT-video basename) |
| clip order | `--order-csv <csv>` → ordered by its `video` column basename; else sorted filename |
| clips used | the **intersection** present in *every* run dir (so side-by-side stays aligned); skipped ones are logged |
| panel layout | runs placed left→right in the order given; panels resized to equal height before hstack |
| overlay | `NNN  <clip-filename>` top-left (yellow on black), per clip; toggle with `--no-label` |

Each eval clip is usually already vertically stacked (model output over its GT); the
tool keeps that as-is, so a 2-run comparison shows `[A-out/A-GT] | [B-out/B-GT]`.

## How to run it (always via the launcher, in the background)

```bash
.claude/skills/concat-eval-videos/launch_concat.sh -- \
  --runs output/<runA>-<step>-<set> output/<runB>-<step>-<set> \
  --labels <shortA>,<shortB> \
  --order-csv data/.../<test_set>.csv \
  --out-dir output/<compare_dir> \
  --out-prefix compare
```

Everything after `--` is forwarded to `concat_videos.py`. Launcher knobs (env or
leading flags): `--qos`/`QOS` (default `boost_qos_dbg`), `--time`/`TIME`
(`00:30:00`), `ACCOUNT`, `PARTITION`, `CPUS`, `MEM`, `WORKSPACE` (repo root, default
cwd), `SIF`. The compositor blocks (model-free but IO-bound), so the **agent should
launch it as a background Bash command** (`run_in_background: true`) and report
completion, then verify a frame.

### concat_videos.py flags

- `--runs DIR...` (required) — one or more run output dirs; each becomes a panel left→right.
- `--labels A,B` — short names used in output filenames (default: dir basenames).
- `--order-csv CSV` / `--csv-col COL` — order clips by CSV column basename (default col `video`); omit to sort by filename.
- `--out-dir`, `--out-prefix` — where to write / comparison basename prefix.
- `--fps` (default 25).
- `--no-label` — drop the index+filename overlay (on by default).
- `--no-comparison` — only write per-run concat videos, skip the side-by-side.

## Outputs

- `<out-dir>/<label>-concat.mp4` — one per run.
- `<out-dir>/<prefix>_<labelA>_vs_<labelB>..._lr.mp4` — the side-by-side (only when ≥2 runs).

## Procedure for the agent

1. Identify the run output dirs (the `--save_path` of each eval, e.g. the
   `eval-training` `--launch` jobs write `output/<run>-<step>-<set>/`). Confirm each
   holds per-clip `*.mp4`.
2. Pick the clip order: pass the same `--order-csv` the evals used so both runs align
   by clip; otherwise filename sort is fine when both dirs hold the same names.
3. Launch via `launch_concat.sh -- <args>` as a **background** Bash command; report
   the Slurm job + output dir.
4. On completion, verify: extract one comparison frame (in-container) and Read it to
   confirm layout + overlay; clean up any temp PNGs.

## Gotchas

- **Container only** — see above; the launcher binds `/workspace`, `/workspace/models`,
  `/workspace/outputs` like the other skills.
- **Alignment** — only clips present in *all* runs are used; a run missing clips
  shrinks the comparison. The skip list is logged.
- **Audio** — eval clips are typically silent (the eval's `merge_video_audio` needs a
  system `ffmpeg` the container lacks); the concat/comparison are silent too.
- **`nframes=inf`** — mp4 metadata often misreports frame count; the tool streams with
  `zip` over readers and a `MAX_FRAMES` cap rather than materializing frame lists
  (materializing was what OOM'd the original one-off).
- **Equal heights** — panels are resized to the first run's height before hstack, so
  runs at different resolutions still compose.
