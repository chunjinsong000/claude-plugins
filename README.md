# Claude Code Custom Skills

Custom [Claude Code](https://claude.com/claude-code) skills for working with
**Wan2.2-S2V LoRA training** and **SLURM** on an HPC cluster.

## Skills

| Skill | What it does |
|-------|--------------|
| [`setup-slurm`](setup-slurm/) | Install and configure a single-node SLURM cluster on Ubuntu with Gmail-relayed email notifications for job `BEGIN`/`END`/`FAIL`. Auto-detects CPUs/memory/NVIDIA GPUs (exposes GPUs as `gres`), uses `proctrack/linuxproc` for cgroup-v2 hosts, and wires `msmtp` → Gmail (handling the AppArmor logfile path and `slurm`-user readable `msmtprc` gotchas). Runs an end-to-end test job that fires the emails. |
| [`continue-training`](continue-training/) | Resume a Wan2.2-S2V LoRA run from its latest (or chosen) checkpoint. Generates a new `..._leo_<N+1>.sh` SLURM script that resumes via `--lora_checkpoint` with a configurable `--skip_frames`, following the project's `_cont<STEP>_SF<SF>` convention. |
| [`eval-training`](eval-training/) | Generate the `livedealer_infer.py` evaluation command for a training run (script or SLURM job id). Emits the inference snippet on the eyes-only test set with the correct `WAN_*` env flags, `lora_path`/step, width/height, and pose/object inputs. |
| [`self-forcing`](self-forcing/) | Launch, tune, and monitor the Self-Forcing experiments for Wan2.2-S2V-14B (`task=sft`/`dmd_distill` + `--self_forcing`) from `live_dealer/train/self_forcing/`. Picks the right launcher per box (B200 SLURM+singularity vs bare-metal 2xH100), validates the `NUM_FRAMES` block geometry (`4*fpb*k+1`, >=2 blocks) before launch, applies knobs via env overrides, and knows the per-variant memory dials (per-block vs per-denoising-step KV caches), the checkpoint/resume shapes, and the DMD student-LoRA deployment gotcha. |
| [`benchmark-streaming`](benchmark-streaming/) | Run a real-streaming S2V inference **performance/telemetry benchmark** (the streaming pipeline with `--profile --gpu_telemetry`, not the VAE-only isolation mode) and analyze it. `run_bench.py` mirrors `livedealer_infer_real_streaming_lambda.sh`, parametrizes the perf knobs (resolution / fps / steps / GPU count / frames-per-block / KV-cache) into a self-clobbering `output/bench/...` dir, and can launch it. `summarize_bench.py` reads the `gpu_telemetry.csv` + `module_fps.csv` (+ optional `memory_report.txt`) it drops, renders `gpu_curves.png` per run via the repo plotter, and prints a per-run + cross-run markdown comparison table (steady-state DiT / VAE-decode / pose fps, mean/peak power/temp/clock, peak device memory). |
| [`slurm-wait-analysis`](slurm-wait-analysis/) | Query `sacct` for your jobs, compute queue wait time and run time, and write a readable Markdown/HTML report with a per-job table, summary cards, and averages. Supports filtering by node count and minimum run time. |
| [`transfer-lambda-to-leonardo`](transfer-lambda-to-leonardo/) | Copy/sync data from a Lambda cloud instance (or any non-LEONARDO Linux box) to LEONARDO `$WORK`. Solves the no-smallstep-cert auth problem via SSH agent forwarding, creates the destination dir (remote rsync is 3.1.3, no `--mkpath`), and runs a resumable rsync to the datamover with a `dmover1-4` parallel-split option for many small files. |
| [`transfer-lambda-to-lambda`](transfer-lambda-to-lambda/) | Copy/sync data between two Lambda cloud instances (or any two Linux boxes sharing one SSH keypair). Solves the source-can't-reach-target hop via SSH agent forwarding (never copies the private key), creates the dest dir (`--mkpath`), and runs a resumable rsync with a parallel-streams option for many small files. |
| [`LTX_experiment`](LTX_experiment/) | Set up and run **LTX-2.3 LipDub** video dubbing — re-dub the speech in an existing clip to new dialogue while preserving the speaker's face/scene/motion (IC-LoRA on the LTX-2.3-22B distilled model). Clones the [LTX-2](https://github.com/Lightricks/LTX-2) code, documents the Hugging Face gate/auth requirement, downloads the gated weights (base + spatial upscaler + LipDub IC-LoRA + Gemma text encoder) via `download_lipdub_models.sh`, and runs the verified `ltx_pipelines.lipdub` CLI via `run_lipdub.sh` (frame count/fps taken from the reference video; low-VRAM `fp8-cast`/offload option). |

## Installation

Each skill is a directory containing a `SKILL.md` (with name/description
frontmatter) plus its supporting scripts. To use them in a project, copy or
symlink the skill directories into that project's `.claude/skills/`:

```bash
git clone https://github.com/<your-username>/claude-skills.git
ln -s "$(pwd)/claude-skills/continue-training"   /path/to/project/.claude/skills/continue-training
ln -s "$(pwd)/claude-skills/eval-training"       /path/to/project/.claude/skills/eval-training
ln -s "$(pwd)/claude-skills/slurm-wait-analysis" /path/to/project/.claude/skills/slurm-wait-analysis
```

Or install them user-wide under `~/.claude/skills/`.

Claude Code discovers each skill from its `SKILL.md` and invokes it when your
request matches the skill's description.
