# diffsynth — Wan2.2-S2V Training & Eval

Skills for **Wan2.2-S2V LoRA / self-forcing training and evaluation** in the
DiffSynth repo (live_dealer pipeline). Machine-level workflows; the DiffSynth
repo itself carries additional repo-specific skills under its own `.claude/skills/`.

## Skills

| Skill | What it does |
|-------|--------------|
| [`self-forcing`](skills/self-forcing/) | Launch, tune, and monitor the Self-Forcing experiments for Wan2.2-S2V-14B (`task=sft`/`dmd_distill` + `--self_forcing`) from `live_dealer/train/self_forcing/`. Picks the right launcher per box (B200 SLURM+singularity vs bare-metal 2xH100), validates the `NUM_FRAMES` block geometry (`4*fpb*k+1`, >=2 blocks) before launch, applies knobs via env overrides, and knows the per-variant memory dials (per-block vs per-denoising-step KV caches), the checkpoint/resume shapes, and the DMD student-LoRA deployment gotcha. |
| [`continue-training`](skills/continue-training/) | Resume a Wan2.2-S2V LoRA run from its latest (or chosen) checkpoint. Generates a new `..._leo_<N+1>.sh` SLURM script that resumes via `--lora_checkpoint` with a configurable `--skip_frames`, following the project's `_cont<STEP>_SF<SF>` convention. |
| [`eval-training`](skills/eval-training/) | Generate the `livedealer_infer.py` evaluation command for a training run (script or SLURM job id). Emits the inference snippet on the eyes-only test set with the correct `WAN_*` env flags, `lora_path`/step, width/height, and pose/object inputs. |
| [`eval-self-forcing`](skills/eval-self-forcing/) | Evaluate a self-forcing (causal, few-step) S2V run against a bidirectional/SFT baseline eval of the same test set: GT \| baseline \| self-forcing 3-panel comparison video, face-centred head-zoom comparison, and a temporal-consistency matrix (warp error, tOF, temporal LPIPS, static-background flicker, PSNR drift) with charts. |
| [`benchmark-streaming`](skills/benchmark-streaming/) | Run a real-streaming S2V inference **performance/telemetry benchmark** (the streaming pipeline with `--profile --gpu_telemetry`, not the VAE-only isolation mode) and analyze it. `run_bench.py` mirrors `livedealer_infer_real_streaming_lambda.sh`, parametrizes the perf knobs (resolution / fps / steps / GPU count / frames-per-block / KV-cache) into a self-clobbering `output/bench/...` dir. `summarize_bench.py` reads the telemetry CSVs, renders `gpu_curves.png` per run, and prints per-run + cross-run markdown comparison tables. |
| [`compare-eval-results`](skills/compare-eval-results/) | Generate an HTML visualization summary comparing two card-detection evaluation results against ground truth: side-by-side wrong-detection frames (pred A \| pred B \| GT) and wrong-card position analysis. |
| [`concat-eval-videos`](skills/concat-eval-videos/) | Concatenate per-clip eval output videos into one timeline per run, and optionally build a side-by-side comparison of two or more runs (clip index + filename overlay), keeping only clips common to all runs so panels stay frame-aligned. |
