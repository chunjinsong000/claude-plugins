# lambda-gpu — Lambda GPU Machine Infra

Infrastructure skills for Lambda.ai GPU boxes (and similar bare-metal Ubuntu GPU hosts).

## Skills

| Skill | What it does |
|-------|--------------|
| [`setup-slurm`](skills/setup-slurm/) | Install and configure a single-node SLURM cluster on Ubuntu with Gmail-relayed email notifications for job `BEGIN`/`END`/`FAIL`. Auto-detects CPUs/memory/NVIDIA GPUs (exposes GPUs as `gres`), uses `proctrack/linuxproc` for cgroup-v2 hosts, and wires `msmtp` → Gmail (handling the AppArmor logfile path and `slurm`-user readable `msmtprc` gotchas). Runs an end-to-end test job that fires the emails. |
| [`slurm-wait-analysis`](skills/slurm-wait-analysis/) | Query `sacct` for your jobs, compute queue wait time and run time, and write a readable Markdown/HTML report with a per-job table, summary cards, and averages. Supports filtering by node count and minimum run time. |
| [`transfer-lambda-to-leonardo`](skills/transfer-lambda-to-leonardo/) | Copy/sync data from a Lambda cloud instance (or any non-LEONARDO Linux box) to LEONARDO `$WORK`. Solves the no-smallstep-cert auth problem via SSH agent forwarding, creates the destination dir (remote rsync is 3.1.3, no `--mkpath`), and runs a resumable rsync to the datamover with a `dmover1-4` parallel-split option for many small files. |
| [`transfer-lambda-to-lambda`](skills/transfer-lambda-to-lambda/) | Copy/sync data between two Lambda cloud instances (or any two Linux boxes sharing one SSH keypair). Solves the source-can't-reach-target hop via SSH agent forwarding (never copies the private key), creates the dest dir (`--mkpath`), and runs a resumable rsync with a parallel-streams option for many small files. |
