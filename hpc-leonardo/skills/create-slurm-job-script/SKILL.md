---
name: create-slurm-job-script
description: Use when the user wants to create a new SLURM training sbatch script on LEONARDO ("create training script", "new training job", "submit a training run", "起一个训练 SLURM 脚本"). Default policy for inference is to submit a short boost_qos_dbg job (highest priority, ≤30 min, fast queue); only run inference interactively on an allocated compute node when the user explicitly wants live iteration.
---

# LEONARDO Create Training Job Script

## Overview

Generate a new SLURM sbatch script by copying one of the canonical templates
under `assets/` and substituting placeholders. Two variants exist:

- `assets/slurm_template_multinode_train.sh` — **CONDA (default)**.
  Activates a conda env directly on the compute node. Use for typical
  training workflows; lighter, no container overhead.
- `assets/slurm_template_multinode_train_sif.sh` — **SIF (Singularity)**.
  Runs inside a pre-built container image. Use when the project needs the
  reproducibility/CUDA-forward-compat baked into the SIF (default image:
  `$WORK/shared/singularity/diffsynth-a100/diffsynth-bind-a100.sif`).

Cluster facts (QoS, account naming, NCCL settings, hardware) live in
`../../references/slurm.md` and `../../references/hardware.md` — this SKILL.md only
encodes the **procedure** and the workflow-specific judgment calls (env
detection, naming, mistakes-to-avoid).

**Inference default policy**: submit inference as a short `boost_qos_dbg`
job (highest priority, ≤ 30 min walltime, fast queue — see § QoS
Selection). Debug QoS schedules fastest and the job self-terminates at
walltime, so it's the cleanest way to run a bounded inference pass
without holding an interactive allocation open. Generate an sbatch
inference script from the same templates below; only the entry-point
script and the `--qos=boost_qos_dbg --time=00:30:00` directives differ
from a training script. Submission stays the user's call (you produce
the script, they run `sbatch`).

**When to run inference interactively instead**: only when the user
*explicitly* wants live iteration — a REPL-style session, debugging a
model load, or repeated manual prompts where re-queuing a job each time
is wasteful. In that case ssh into a compute node from an existing SLURM
allocation (NOT `leo` login nodes, which have no GPU), or use
`~/scripts/1gpu.sh` / `1node.sh` to grab an interactive debug allocation.
Confirm this is what they want before going interactive — don't assume.

**When inference needs > 30 min** (large sweep over many ckpts, long
video generation): use `normal` (≤ 24 h) instead of `boost_qos_dbg`, and
tell the user why the QoS differs from the debug default.

## QoS Selection (single source of truth)

Account `AIFAC_F02_378` has 4 QoS on partition `boost_usr_prod`. Always
pick the **tightest** QoS that fits the user's `(walltime, nodes)` — it
schedules faster (less restrictive QoS = bigger queue ahead).

| QoS | Max walltime | Max nodes | Max jobs (run/submit) | Priority | `--switches` (topology hint) | When to pick |
|---|---|---|---|---|---|---|
| `boost_qos_dbg` | 00:30:00 | 8 | **2 / 2** | 80 (highest) | `2@00:10:00` | Code debugging, smoke tests, quick iteration |
| `normal` | 1-00:00:00 (24 h) | partition default | unlimited | 40 | `2@01:00:00` | Standard production within 1 day; convention nodes ≤ 8 |
| `boost_qos_bprod` | 1-00:00:00 (24 h) | 256 (min 65) | unlimited | 60 | `2@01:00:00` | **Big** production: same walltime as `normal`, scale to many nodes |
| `boost_qos_lprod` | 4-00:00:00 (4 days) | 8 | unlimited | 40 | `2@01:00:00` | **Long** production: > 24 h, ≤ 8 nodes |

`--switches=count@max-wait` asks SLURM to schedule within ≤ `count` IB
leaf-switches, waiting up to `max-wait` for tight allocation. Pair with
QoS walltime: 10-min wait for 30-min debug jobs, 1-hour wait for 24-h+
production. Tight topology reduces inter-node hop count for NCCL.

**`boost_qos_dbg` job-count cap**: per user, at most **2 jobs** existing
at once (running + pending combined — `MaxJobsPU=2`, `MaxSubmitPU=2`),
and their node total must be ≤ 8. Submitting a 3rd is rejected with
`sbatch: error: QOSMaxSubmitJobPerUserLimit`. The other three QoS have no
per-user job-count cap (only the node/walltime caps above).

**Priority** (column above): higher number = scheduled sooner. It is the
QoS contribution to the multifactor priority (weight 300000 on Leonardo,
the single biggest lever — far above age=20000 / fairshare=25000). So
`boost_qos_dbg` (80) clears the queue fastest; `normal` and
`boost_qos_lprod` (both 40) wait longest. When a `normal` job is stuck
behind hundreds of pending jobs, switching it to `boost_qos_bprod` (60)
via `scontrol update jobid=<id> qos=boost_qos_bprod` can leapfrog most of
them (no per-account cap on bprod).

**Auto-pick algorithm** (apply in order; return first match):

1. walltime ≤ 30 min  → `boost_qos_dbg`
2. walltime ≤ 24 h AND nodes ≤ 8 → `normal`
3. walltime ≤ 24 h AND nodes > 8 → `boost_qos_bprod`
4. 24 h < walltime ≤ 4 days AND nodes ≤ 8 → `boost_qos_lprod`
5. walltime > 4 days OR (walltime > 24 h AND nodes > 8) → **stop**: ask
   user to split into checkpoints; no single QoS covers it.

When the user gives you walltime + nodes but no QoS, auto-pick from this
table and explicitly report which one and why ("walltime 2h, 4 nodes →
`normal`"). When the user gives you a QoS that contradicts walltime/nodes
(e.g. `normal --time=4-00:00:00`), STOP and surface the conflict.

## Steps

1. **Confirm intent: training or inference**. Both produce an sbatch
   script via this skill. For inference, apply the inference default
   policy above — generate a `boost_qos_dbg` sbatch script (≤ 30 min) by
   default, switching to `normal` only if it needs > 30 min, or going
   interactive only if the user explicitly wants live iteration.

2. **Gather these inputs from the user** (ASK if any is missing — do
   NOT guess; missing inputs are the #1 source of skill failures):
   - **Execution mode**: `conda` (default) or `sif`. Pick the matching
     template under `assets/` accordingly.
   - Project directory (`$WORK/$USER/<dir>` — see `../../references/storage-layout.md` § 3)
   - Training entry-point (relative to project dir, e.g. `examples/.../train.py`).
     For SIF mode, the path lives under `/workspace/...` from the
     container's view.
   - Number of nodes — whole-node only; powers of 2 (1, 2, 4, 8, 16, 32, 64)
   - Walltime (`HH:MM:SS` or `D-HH:MM:SS`)
   - QoS — if user doesn't say, **auto-pick from § QoS Selection** above
     using `(walltime, nodes)`. Report the choice + reason back to them.
   - **`TODO_scratch_project_dir`** — absolute path to `$SCRATCH/<project>`
     (same basename as `TODO_project_dir`). Slurm logs, outputs, and runs
     all land here. Don't guess; if uncertain, default to
     `/leonardo_scratch/large/userexternal/$USER/<basename of project_dir>`
     and ASK to confirm.
   - **`TODO_dataset_base`** — absolute host path to the dataset root.
     Don't guess from project name. ASK if user didn't say.
   - Output dir: defaults to `$SCRATCH_PROJECT_DIR/outputs/job_${SLURM_JOB_ID}`
     inside the template; no `TODO_output_dir` to fill. User can `export
     OUTPUT_DIR=<custom path>` before sbatch if they need a non-default
     location.
   - **`TODO_ds_config`** (conda mode) or **`TODO_ds_config_in_container`**
     (SIF mode) — absolute path to DeepSpeed config JSON. SIF path uses
     the container view (e.g. `/workspace/...`). If project doesn't use
     DeepSpeed, ASK whether to drop the `--deepspeed_config_file` flags
     entirely. Don't pick a default.
   - Compute account stays `AIFAC_F02_378` unless the user explicitly says
     otherwise (see `../../references/slurm.md` § 1).

3. **Detect conda env name** from the project dir basename (rules below
   in "Conda Env Detection Rule"). If unsure, run
   `ls /leonardo_work/AIFAC_F02_378/$USER/conda/envs/` and ask user to confirm.

4. **Build the job name** following the convention below ("Job Name
   Convention"). Always end with `_<N>n` so node count is visible in
   `squeue` output.

5. **Read the canonical template** matching the execution mode from step 2:
   - `conda` mode → `~/.claude/skills/create-slurm-job-script/assets/slurm_template_multinode_train.sh`
   - `sif` mode → `~/.claude/skills/create-slurm-job-script/assets/slurm_template_multinode_train_sif.sh`

   Do NOT modify the templates in place. Do NOT copy a stale older version.

6. **Copy to project**:
   ```bash
   mkdir -p <project_dir>/train
   # conda mode (default):
   cp ~/.claude/skills/create-slurm-job-script/assets/slurm_template_multinode_train.sh \
      <project_dir>/train/<job_name>.sh
   # sif mode:
   # cp ~/.claude/skills/create-slurm-job-script/assets/slurm_template_multinode_train_sif.sh \
   #    <project_dir>/train/<job_name>.sh
   ```

   Storage convention:
   - **Code** on `$WORK/$USER/<project>/` (small, backed up).
   - **`slurm_logs/`** is **user-global** at `$SCRATCH/slurm_logs/` —
     all jobs from all projects share the same dir, distinguished by
     `%x_%j` (jobname_jobid) in filename.
   - **`outputs/` / `runs/`** are **per-project** under `$SCRATCH/<project>/`.
   - Work-side `outputs/` and `runs/` are **symlinks** → the scratch dirs.

   One-time setup:
   ```bash
   # User-global slurm_logs (one-time, ever)
   mkdir -p /leonardo_scratch/large/userexternal/$USER/slurm_logs

   # Per-project setup
   PROJECT_NAME=<project>   # same basename used under $WORK/$USER/
   SCRATCH_PROJECT_DIR=/leonardo_scratch/large/userexternal/$USER/$PROJECT_NAME
   WORK_PROJECT_DIR=/leonardo_work/AIFAC_F02_378/$USER/$PROJECT_NAME

   # 1) Create scratch-side per-project dirs
   mkdir -p "$SCRATCH_PROJECT_DIR"/{outputs,runs}

   # 2) Work-side runs/ and outputs/ → symlinks into scratch
   [ -e "$WORK_PROJECT_DIR/outputs" ] || ln -s "$SCRATCH_PROJECT_DIR/outputs" "$WORK_PROJECT_DIR/outputs"
   [ -e "$WORK_PROJECT_DIR/runs"    ] || ln -s "$SCRATCH_PROJECT_DIR/runs"    "$WORK_PROJECT_DIR/runs"
   ```
   Add `outputs`, `runs` to the project's `.gitignore` (the work-side
   symlinks shouldn't be tracked; real data lives on scratch).

7. **Substitute placeholders.** The contract is one-way: **every
   fillable field in the template is a `TODO_*` token; every non-TODO
   value is a measured/validated invariant — don't touch it.** Apply the
   substitutions one at a time using whatever file-editing tool your
   harness exposes; one substitution per change so each is verifiable.
   Mapping:

   | Token in template | Replace with |
   |---|---|
   | `TODO_jobname` | `<job_name>` from step 4 |
   | `TODO_nodes` | `<N>` from step 2 (1, 2, 4, 8, 16, ...) |
   | `TODO_account` | `AIFAC_F02_378` (default; only change if user explicitly says the compute account has rotated) |
   | `TODO_qos` | QoS chosen in step 2 (`boost_qos_dbg` / `normal` / `boost_qos_bprod` / `boost_qos_lprod`) — see § QoS Selection |
   | `TODO_time` | the walltime from step 2 (must respect the QoS cap from § QoS Selection) |
   | `TODO_switches` | `2@00:10:00` for `boost_qos_dbg`; `2@01:00:00` for `normal` / `boost_qos_bprod` / `boost_qos_lprod` (see § QoS Selection) |
   | `TODO_project_dir` | absolute path to **code** under `$WORK/$USER/<project>` (e.g. `/leonardo_work/AIFAC_F02_378/$USER/diffsynth2`) |
   | `TODO_scratch_project_dir` | absolute path to **scratch project dir** `$SCRATCH/<project>` (same project basename; e.g. `/leonardo_scratch/large/userexternal/$USER/diffsynth2`). Templates write `slurm_logs/`, `outputs/`, `runs/` under here. |
   | `TODO_dataset_base` | absolute host path to the dataset root (usually under `$SCRATCH/...`; can be team-shared) |
   | `TODO_conda_env` (conda mode only) | full env path (e.g. `/leonardo_work/AIFAC_F02_378/$USER/conda/envs/diffsynth2`) or env name from step 3 |
   | `TODO_ds_config` (conda mode) | absolute host path to the project's `ds_config*.json` (DeepSpeed); if no DeepSpeed: comment this line AND remove `--deepspeed_config_file $DS_CONFIG` + `--use_deepspeed` from the `accelerate launch` block |
   | `TODO_ds_config_in_container` (SIF mode) | container-view path to `ds_config*.json` (project files are bound at `/workspace`, so this typically starts with `/workspace/...`) |
   | `TODO_path/to/train_script.py` (conda mode) | entry-point relative to `$PROJECT_DIR` |
   | `TODO_path_in_container/train_script.py` (SIF mode) | container-view path, usually `/workspace/...` |
   | `TODO_train_script_flags` | the script's own CLI args (lr, dataset paths, model config, etc.). **No defaults are shipped** — ASK the user for project-specific flags or leave the placeholder so they fill it in. |

   `WAN_MODELS=` / `MODELSCOPE_CACHE=$WAN_MODELS` (conda) and the `--env WAN_MODELS=/wan_models`
   / `--env MODELSCOPE_CACHE=/wan_models` lines (SIF) are only relevant for Wan-based projects.
   For non-Wan projects, comment those lines.

8. **Verify** before reporting completion:
   - `bash -n <new_script>` — syntax must be clean.
   - `grep -n 'TODO_' <new_script>` — **canonical contract; must print
     nothing**. If it prints anything, you missed a placeholder. This is
     the single source of truth for "all fillable fields are filled" —
     it does not depend on the template's other defaults staying stable.
   - Spot-check the directives the user explicitly specified
     (`grep -E '^#SBATCH --(nodes|qos|time)=' <new_script>`) match
     what they asked for.

9. **Show the user**:
   - The full path to the new script.
   - A summary: nodes / QoS / walltime / account.
   - **For `normal` jobs**: show this submit-and-track one-liner so
     the user can submit AND auto-add a row to `runs/RUNNING.tsv` in
     one shot. Format spec: `../../references/runs-tsv-format.md`.
     ```bash
     SCRIPT='<absolute-path-to-script>'

     mkdir -p runs
     [ -f runs/RUNNING.tsv ] || printf '%s\n' \
         $'JobID\tScript\tQoS\tNodes\tTime\tSubmitted\tState\tStep\tLoss\tLastChecked' \
         > runs/RUNNING.tsv

     JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
     QOS=$(grep -oP '^#SBATCH --qos=\K\S+'   "$SCRIPT")
     N=$(  grep -oP '^#SBATCH --nodes=\K\d+' "$SCRIPT")
     T=$(  grep -oP '^#SBATCH --time=\K\S+'  "$SCRIPT")
     NOW=$(date '+%Y-%m-%d %H:%M')
     printf '%s\t%s\t%s\t%s\t%s\t%s\tPENDING\t-\t-\t%s\n' \
         "$JOB_ID" "$(basename "$SCRIPT")" "$QOS" "$N" "$T" "$NOW" "$NOW" \
         >> runs/RUNNING.tsv

     echo "Submitted $JOB_ID, tracked in runs/RUNNING.tsv"
     ```
     The QoS/N/Time fields are auto-extracted from the script via grep
     — no manual substitution needed.
   - **For `boost_qos_dbg` jobs**: just give plain `sbatch <path>`. Don't
     track in RUNNING.tsv (RUNNING.tsv is for production only —
     `boost_qos_dbg` is too noisy).
   - DO NOT run `sbatch` yourself. Provide the command; user runs it.

## Don't Modify These Parts of the Template

When substituting placeholders, do not touch the rest of the template.
Each block below is measured/validated; the *why* is in `../../references/`.

| Block | Reference |
|---|---|
| NCCL settings on Leonardo Booster (pure IB: `ib` / `mlx5` prefix match, `NCCL_IB_GID_INDEX=0`, `NCCL_PXN_DISABLE=1`, `NCCL_NVLS_ENABLE=0`, `NCCL_NET_GDR_LEVEL=PHB`, etc.) | `../../references/hardware.md` § NCCL |
| `SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}` | `../../references/slurm.md` § 8 |
| Whole-node directives (`--gres=gpu:4`, `--cpus-per-task=32`) | `../../references/slurm.md` § 6 |
| Log paths `/leonardo_scratch/large/userexternal/%u/slurm_logs/%x_%j.{out,err}` (user-global on scratch, NOT per-project; MUST `mkdir -p` once before any sbatch) | (template) |
| Cache redirects (`HF_HOME=$TMPDIR`, `TRITON_CACHE_DIR=$TMPDIR`, etc.) — Leonardo `$TMPDIR` is 10 GB tmpfs | template comments |
| `MODELSCOPE_CACHE=$WAN_MODELS` (NOT `$TMPDIR`) — only for Wan projects | template comments |
| `CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.6/none` + `PATH` (conda mode only) — for DeepSpeed's nvcc version check. Do NOT touch `LD_LIBRARY_PATH` here. | (template) |
| Master discovery `MASTER_ADDR=$(scontrol show hostnames ... \| head -n 1)` (Leonardo: hostname is fine; NCCL bootstrap is ~KBs of KV exchange then RDMA over IB) | `../../references/slurm.md` + onboarding `slurm-env-reference.md` |
| SIF bind mounts (`/workspace`, `/wan_models`, `/data`, `/output`, `/tmp`) (SIF mode only) | template comments |

## Conda Env Detection Rule

Working directory basename → env name:
- Lowercase the basename: `DiffSynth-valka` → `diffsynth-valka`
- **Trailing digits MUST be preserved**: `DiffSynth1` → `diffsynth1` (NOT `diffsynth`)

| Project dir basename | Correct env name | Wrong (don't use) |
|---|---|---|
| `diffsynth2` | `diffsynth2` | `diffsynth` |
| `DiffSynth-valka` | `diffsynth-valka` | `diffsynth_valka` |
| `DiffSynth1` | `diffsynth1` | `diffsynth` |
| `DiffSynth1-wan2-2-vae-lora` | `diffsynth1-wan2-2-vae-lora` | `diffsynth-wan2-2-vae-lora` |
| `LiveAvatar` | `liveavatar` | `live-avatar` |
| `MatAnyone` | `matanyone` | `matanyone1` |

If `ls /leonardo_work/AIFAC_F02_378/$USER/conda/envs/` doesn't show your
inferred name, ask the user. (Conda base on Leonardo lives at
`/leonardo_work/AIFAC_F02_378/$USER/conda/miniforge3`; envs sit at
`/leonardo_work/AIFAC_F02_378/$USER/conda/envs/<name>`.)

## Job Name Convention

Format: `<task>_<config>?_<N>n`

- Always end with `_<N>n` (node count visible in `squeue`).
- snake_case, all lowercase, under 30 chars, no spaces or special chars.

Examples:
- `s2v_lora_4n` — speech-to-video LoRA, 4 nodes
- `wan22vae_stage1_8n` — Wan 2.2 VAE stage 1, 8 nodes
- `wan22_lora_lr5e5_2n` — LoRA with lr=5e-5, 2 nodes
- `train_quick_1n` — smoke test, 1 node

## Common Mistakes — DO NOT MAKE THESE

| Mistake | Right way |
|---|---|
| Filling `TODO_account` with a non-`AIFAC_F02_378` value without being told to | Default to `AIFAC_F02_378`; user will say if the compute account has rotated. See `../../references/slurm.md` § 1 |
| `--gres=gpu:2` for "small test" | Always `--gres=gpu:4`; use `--nodes=1` for small tests |
| `normal --time=00:30:00` for a quick test | Use `boost_qos_dbg` for tests; `normal` only for real training |
| `conda activate diffsynth` for `diffsynth2` dir | env name MUST match trailing digit: `diffsynth2` |
| Removing the NCCL block "for cleanliness" | All lines are measured/validated; keep every one |
| Removing the `$TMPDIR` cache block | Required to avoid Lustre metadata storms and JIT lock contention |
| Adding `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH` in conda mode | NEVER. That dir ships `libnvidia-ml.so` 560.35 which mismatches Leonardo's host driver → `Failed to initialize NVML: Driver/library version mismatch`. CUDA_HOME + PATH is enough. |
| Auto-running `sbatch <new_script>` | NEVER. User reviews, user submits |

## Red Flags — STOP and reconsider

If you find yourself thinking any of these, STOP and re-read this skill:

- "I'll edit NCCL settings to optimize"
- "User asked for 2 GPUs, let me set --gres=gpu:2"
- "User didn't tell me AIFAC_F02_378, but I see another account name floating around — let me use that" (NO. Default to AIFAC_F02_378 unless explicitly told otherwise)
- "User said quick test → I'll use normal with short --time"
- "I'll skip the cache block to reduce script length"
- "User didn't give a dataset path — let me invent one based on the project name" (NO. ASK the user. Skill explicitly forbids guessing TODO_dataset_base / TODO_output_dir / TODO_ds_config.)
- "I'll add a `module load` line — that's the BSC way" (NO. Leonardo conda template uses the conda env directly; SIF template runs inside a container. No `module load` is needed in either.)
- "I'll prepend `$CUDA_HOME/lib64` to LD_LIBRARY_PATH so torch finds the right libs" (NO. Causes NVML driver/library mismatch on Leonardo. CUDA_HOME + PATH is enough.)
- "DiffSynth1 dir → diffsynth env should work"
- "Let me sbatch it for the user, they're in a hurry"

All violations of invariants. Don't.

## What this skill does NOT do

- Submitting the job (`sbatch` is always the user's call)
- Modifying the canonical templates under `assets/` (the conda one and the SIF one)
- Data preprocessing scripts
- Benchmark or profiling scripts

## Background References

For the *why* behind any directive (read on demand, not eagerly):
- `../../references/hardware.md` — A100 / NVLink / NCCL fabric / IP-vs-RDMA
- `../../references/slurm.md` — QoS, accounts, switches, CPU rule, logs, SRUN
- `../../references/storage-layout.md` — projects/ vs scratch/, `${USER}`/`%u` portability rule
- `../../references/lustre-resilience.md` — D-state, probe pattern, why warmup matters
