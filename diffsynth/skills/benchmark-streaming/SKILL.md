---
name: benchmark-streaming
description: Run a real-streaming S2V inference performance/telemetry benchmark and analyze the results. Launches live_dealer/infer/livedealer_infer_real_streaming.py (the streaming pipeline — NOT the vae-only isolation mode) with --profile + --gpu_telemetry so it drops gpu_telemetry.csv (core-clock/power/temp) and module_fps.csv (per-module fps) into the save dir, optionally sweeping perf knobs (resolution, fps, steps, GPU count, frames/block, KV-cache). Then analyzes those CSVs into gpu_curves.png plots and a per-run + cross-run markdown comparison table (steady-state DiT / VAE-decode / pose fps, mean/peak power/temp/clock, peak device memory). Use when the user says "benchmark the streaming inference", "profile the streaming run", "measure fps / throughput / GPU usage", "run a perf sweep over resolution/steps", "analyze the telemetry / gpu curves", or "compare the perf of these runs".
---

# Benchmark streaming S2V inference & analyze the telemetry

Two phases: **run** a real-streaming inference benchmark (the streaming pipeline
with profiling + telemetry on), then **analyze** the CSVs it drops into a plot +
comparison table. Scope is the streaming pipeline only — there is no VAE-only
isolation run here.

## What "run" produces

`livedealer_infer_real_streaming.py`, launched with `--profile --gpu_telemetry`,
writes into `<save_path>/`:

| file | written by | contents |
|---|---|---|
| `gpu_telemetry.csv` | `GpuTelemetrySampler` in `live_dealer/infer/utils/log_utils.py` | one row per GPU per `--gpu_telemetry_interval` s: `timestamp,elapsed_s,index,name,clocks_graphics_mhz,clocks_sm_mhz,clocks_mem_mhz,power_draw_w,temperature_gpu_c` |
| `module_fps.csv` | `append_module_fps_csv` | one row per clip: `timestamp,clip_idx,dit_ms,dit_fps,vae_dec_ms,vae_dec_fps,pose_ms,pose_fps,audio_ms` (needs `--profile`; single-GPU rank-0 path) |
| `memory_report.txt` | `report_memory` | per-rank weight/KV/peak breakdown incl. "Peak device used (WHOLE COST)" — only with `--mem_profile` |
| `bench.log` | the launcher (`--launch`) | full stdout/stderr of the run |

> `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`, so `gpu_telemetry.csv` has rows
> for **every** physical GPU — filter by the `index` column (default 1, the
> inference GPU in the lambda scripts).

## Phase 1 — run the benchmark

`run_bench.py` mirrors the env block + torchrun of
`live_dealer/infer/livedealer_infer_real_streaming_lambda.sh`, parametrizes the
perf knobs, and **always forces `--profile --gpu_telemetry`**.

```bash
python3 .claude/skills/benchmark-streaming/run_bench.py \
  [--steps 3] [--cuda-visible-devices 1] \
  [--width 1280 --height 720 --fps 24] \
  [--frames-per-clip 12 --lframes-per-block 3 --lframes-per-kv-cache 3] \
  [--num-clips 25 | --sustained] [--repeat 1] \
  [--lora-path PATH] [--input-image P --pose-video P --audio-path P --card-detection P] \
  [--vae-encoder-type wanvae2.1] [--mem-profile] [--no-warmup] \
  [--save-path DIR] [--launch]
```

- By default it **prints** the env+torchrun snippet (show it to the user) and,
  on the last stdout line, `SAVE_PATH=<dir>` (the auto-derived
  `output/bench/<lora_run>/<step>/<name>` — the `<name>` encodes the config:
  `3bf3kvf_1280-720_24fps_3step_1gpu`, so a sweep never clobbers itself).
- `--launch` runs it (tees to `<save_path>/bench.log`, blocks until done). Model
  load + compile + inference takes minutes, so **invoke the helper with
  `--launch` as a background Bash command** (`run_in_background: true`) and report
  the `SAVE_PATH` + log; surface the result on completion.
- **Bounded vs sustained.** Default is a finite run (`--num-clips 25 --repeat 1`,
  ≈10 s of video) — enough for steady-state fps + a telemetry span. Use
  `--sustained` (`num_clips=1000`, `repeat=0`) only for long thermal / clock
  curves; it loops forever, so stop it with Ctrl-C (or don't background it
  unbounded).
- **GPU count = the partition preset.** `--steps` × the number of GPUs in
  `--cuda-visible-devices` picks the streaming partition (`"<steps>step_<gpus>gpu"`,
  see `README_streaming.md`). Single-GPU (`--cuda-visible-devices 1`) is the
  common perf case and the only one that writes `module_fps.csv`; multi-GPU
  splits write a timing log via `save_log_multi_gpu` instead (parse manually).
- Defaults for lora/image/pose/audio/card come from the talking-mode example in
  the lambda script; override any of them for a different scenario. Resolution
  must be divisible by `2 × vae_downsample` (32 for wanvae2.1) or the CLI rejects it.

### Sweeping

To compare configs, run the helper once per config with the knob varied (each
gets its own `SAVE_PATH`), then hand **all** the save dirs to phase 2. Typical
sweeps: resolution (`--width/--height`), steps (`--steps`), frames-per-block
(`--lframes-per-block`), KV-cache depth (`--lframes-per-kv-cache`), GPU count.

## Phase 2 — analyze the results

`summarize_bench.py` reads one or more run dirs and prints a per-run summary +
a markdown comparison table, and (unless `--no-plot`) renders `gpu_curves.png`
in each dir via the repo's `live_dealer/infer/utils/plot_gpu_curves.py`.

```bash
python3 .claude/skills/benchmark-streaming/summarize_bench.py \
  <run_dir> [<run_dir> ...] \
  [--gpu-index 1] [--warmup-clips 2] [--no-plot] [--markdown bench_compare.md]
```

- **Steady-state** = drop the first `--warmup-clips` rows of `module_fps.csv`
  (compile / cold-cache clips) and take the **median** of the rest. Reports
  DiT / VAE-decode / pose fps (and ms latency).
- Telemetry (filtered to `--gpu-index`, default 1) → mean & peak power (W),
  temperature (°C), core clock (MHz), and the sample span in seconds.
- `memory_report.txt` (if `--mem_profile` was on) → peak device memory (GB).
- Run it from the **repo root** so the plotter path resolves; pass `--no-plot`
  if you only want the table. `--markdown` also writes the table to a file.

## Procedure for the agent

1. Confirm the config the user wants (resolution / steps / GPU count / etc.);
   default to the lambda-script perf case if unspecified.
2. Generate the run with `run_bench.py`. Show the snippet. If they want it
   **run**, re-invoke with `--launch` in the **background** and report
   `SAVE_PATH` + `bench.log`.
3. When the run(s) finish, call `summarize_bench.py` on the save dir(s) and
   present the plot path(s) + comparison table. For a sweep, pass every save dir
   in one call so the table lines them up.

## Notes / gotchas

- This is the **streaming pipeline**, not `--vae_only`. The VAE-only isolation
  test (which writes `vae_fps.csv`) is intentionally out of scope for this skill.
- `module_fps.csv` needs **`--profile`** (run_bench forces it). Without it the
  telemetry CSV still fills but the fps table will be empty.
- Warmup: by default the streaming graphs are compiled up front (`warmup_stream`
  on) so the first emitted clip is already steady-state; `--no-warmup` folds the
  multi-minute compile into clip 0 — bump `--warmup-clips` when summarizing.
- On Lambda the run is a plain `bash` launch (`source .venv/bin/activate`), not
  SLURM/singularity — matching `*_lambda.sh`. For LEONARDO/other hosts, adapt the
  launch wrapper.
