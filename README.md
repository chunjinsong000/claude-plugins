# Claude Code Custom Skills

Custom [Claude Code](https://claude.com/claude-code) skills for working with
**Wan2.2-S2V LoRA training** and **SLURM** on an HPC cluster.

## Skills

| Skill | What it does |
|-------|--------------|
| [`setup-slurm`](setup-slurm/) | Install and configure a single-node SLURM cluster on Ubuntu with Gmail-relayed email notifications for job `BEGIN`/`END`/`FAIL`. Auto-detects CPUs/memory/NVIDIA GPUs (exposes GPUs as `gres`), uses `proctrack/linuxproc` for cgroup-v2 hosts, and wires `msmtp` → Gmail (handling the AppArmor logfile path and `slurm`-user readable `msmtprc` gotchas). Runs an end-to-end test job that fires the emails. |
| [`continue-training`](continue-training/) | Resume a Wan2.2-S2V LoRA run from its latest (or chosen) checkpoint. Generates a new `..._leo_<N+1>.sh` SLURM script that resumes via `--lora_checkpoint` with a configurable `--skip_frames`, following the project's `_cont<STEP>_SF<SF>` convention. |
| [`eval-training`](eval-training/) | Generate the `livedealer_infer.py` evaluation command for a training run (script or SLURM job id). Emits the inference snippet on the eyes-only test set with the correct `WAN_*` env flags, `lora_path`/step, width/height, and pose/object inputs. |
| [`slurm-wait-analysis`](slurm-wait-analysis/) | Query `sacct` for your jobs, compute queue wait time and run time, and write a readable Markdown/HTML report with a per-job table, summary cards, and averages. Supports filtering by node count and minimum run time. |
| [`transfer-lambda-to-leonardo`](transfer-lambda-to-leonardo/) | Copy/sync data from a Lambda cloud instance (or any non-LEONARDO Linux box) to LEONARDO `$WORK`. Solves the no-smallstep-cert auth problem via SSH agent forwarding, creates the destination dir (remote rsync is 3.1.3, no `--mkpath`), and runs a resumable rsync to the datamover with a `dmover1-4` parallel-split option for many small files. |

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
