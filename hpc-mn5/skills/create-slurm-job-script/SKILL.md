---
name: create-slurm-job-script
description: Use when the user wants to create a new SLURM training sbatch script on BSC MareNostrum5 ("create training script", "new training job", "submit a training run", "起一个训练 SLURM 脚本"). Default policy for inference is to run directly on an allocated compute node; only generate an sbatch inference script when the user explicitly asks for batch inference and confirms why.
---

# BSC Create Training Job Script

## Overview

Generate a new SLURM sbatch script by copying the canonical template at
`assets/slurm_template_multinode_train.sh` and substituting placeholders.
Cluster facts (QoS, account naming, NCCL settings, hardware) live in
`../../references/slurm.md` and `../../references/hardware.md` — this SKILL.md only
encodes the **procedure** and the workflow-specific judgment calls (env
detection, naming, mistakes-to-avoid).

**Inference default policy**: inference runs directly on an allocated
compute node (NOT `alogin1` / `alogin2`, which have no GPU), NOT via
sbatch. You either ssh into a compute node from an existing SLURM
allocation, or you are already inside one. This is the default because
most inference is interactive iteration.

**When to generate an sbatch inference script anyway**: only when the
user *explicitly* requests batch inference AND states a reason
(overnight evaluation, queued sweep over many ckpts, etc.). Confirm the
reason before producing one — don't assume "they'll like sbatch better".
If you do generate one, follow the same template and substitution
procedure below; only the entry-point script and walltime will differ.

## Steps

1. **Confirm intent is training**. If the user implies inference: stop
   and apply the inference default policy above (offer to ssh / run on
   the current node). Generate an sbatch inference script only if the
   user explicitly requests it and confirms why.

2. **Gather these inputs from the user** (ASK if any is missing — do
   NOT guess; missing inputs are the #1 source of skill failures):
   - Project directory (`~/projects/<dir>` — see `../../references/storage-layout.md` § 3)
   - Training entry-point (relative to project dir, e.g. `examples/.../train.py`)
   - Number of nodes — must be 1, 2, 4, 8, 16, 32, or 64 (whole-node only;
     see `../../references/slurm.md` § 6)
   - Walltime (`HH:MM:SS` or `D-HH:MM:SS`) — respect the QoS cap below
   - Purpose: code debugging → `acc_debug`; real training → `acc_ehpc`
     (full QoS rules in `../../references/slurm.md` § 2-4)
   - **`TODO_dataset_base`** — absolute path to the dataset root. Don't
     guess from project name. ASK if user didn't say.
   - **`TODO_output_dir`** — absolute path for ckpts/outputs. Don't
     guess. ASK if user didn't say.
   - **`TODO_ds_config`** — absolute path to DeepSpeed config JSON. If
     project doesn't use DeepSpeed, ASK whether to drop the
     `--deepspeed_config_file` flags entirely. Don't pick a default.
   - Compute account stays `ehpc1003` unless Daisy explicitly says
     otherwise (see `../../references/slurm.md` § 1).

3. **Detect conda env name** from the project dir basename (rules below
   in "Conda Env Detection Rule"). If unsure, run
   `ls ~/scratch/envs/` and ask user to confirm.

4. **Build the job name** following the convention below ("Job Name
   Convention"). Always end with `_<N>n` so node count is visible in
   `squeue` output.

5. **Read the canonical template** at
   `assets/slurm_template_multinode_train.sh`.
   Do NOT modify it. Do NOT copy a stale older version.

6. **Copy to project**:
   ```bash
   mkdir -p <project_dir>/train
   cp assets/slurm_template_multinode_train.sh \
      <project_dir>/train/<job_name>.sh
   ```

7. **Substitute placeholders.** The contract is one-way: **every
   fillable field in the template is a `TODO_*` token; every non-TODO
   value is a BSC-validated invariant — don't touch it.** Apply the
   substitutions one at a time using whatever file-editing tool your
   harness exposes; one substitution per change so each is verifiable.
   Mapping:

   | Token in template | Replace with |
   |---|---|
   | `TODO_jobname` | `<job_name>` from step 4 |
   | `TODO_nodes` | `<N>` from step 2 (1, 2, 4, 8, 16, ...) |
   | `TODO_account` | `ehpc1003` (default; only change if user explicitly says the compute account has rotated) |
   | `TODO_qos` | `acc_debug` or `acc_ehpc` from step 2 |
   | `TODO_switches` | `2@0:10:00` if acc_debug; `2@1:00:00` if acc_ehpc |
   | `TODO_time` | the walltime from step 2 (respect the QoS cap) |
   | `TODO_project_dir` | absolute path under `~/projects/<dir>` |
   | `TODO_dataset_base` | absolute path to the dataset root (under `~/scratch/...`) |
   | `TODO_output_dir` | absolute path for ckpts/outputs (under `~/scratch/.../<run_id>`) |
   | `TODO_conda_env` | env name from step 3 |
   | `TODO_ds_config` | absolute path to the project's `ds_config*.json` (DeepSpeed); if no DeepSpeed: comment this line AND remove `--deepspeed_config_file $DS_CONFIG` + `--use_deepspeed` from the `accelerate launch` block |
   | `TODO_path/to/train_script.py` | entry-point relative to `$PROJECT_DIR` |
   | `TODO_train_script_flags` | the script's own CLI args (lr, dataset paths, model config, etc.). **No defaults are shipped** — ASK the user for project-specific flags or leave the placeholder so they fill it in. |

   `WAN_MODELS=` and `MODELSCOPE_CACHE=$WAN_MODELS` are only relevant for
   Wan-based projects. For non-Wan projects, comment those two lines.

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
   - **For `acc_ehpc` jobs**: show this submit-and-track one-liner so
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
   - **For `acc_debug` jobs**: just give plain `sbatch <path>`. Don't
     track in RUNNING.tsv (RUNNING.tsv is for production only —
     `acc_debug` is too noisy).
   - DO NOT run `sbatch` yourself. Provide the command; user runs it.

## Don't Modify These Parts of the Template

When substituting placeholders, do not touch the rest of the template.
Each block below is BSC-validated; the *why* is in `../../references/`.

| Block | Reference |
|---|---|
| NCCL settings (`NCCL_*`) | `../../references/hardware.md` § 4-5 |
| `SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}` | `../../references/slurm.md` § 8 |
| Whole-node directives (`--gres=gpu:4`, `--cpus-per-task=80`) | `../../references/slurm.md` § 6 |
| Log paths (location: `../../references/storage-layout.md` § 6; `%u` syntax: `../../references/slurm.md` § 7) | (multiple) |
| Cache redirects (`HF_HOME=$TMPDIR`, `TRITON_CACHE_DIR=$TMPDIR`, etc.) | template comments |
| `MODELSCOPE_CACHE=$WAN_MODELS` (NOT `$TMPDIR`) | template comments |
| `source $HOME/scripts/job_notifier` (Slack exit notifier) | `~/scripts/job_notifier` |
| `srun --kill-on-bad-exit=1 bash $HOME/scripts/warmup_page_cache.sh ...` (GPFS probe + page-cache warmup) | `../../references/gpfs-hang-resilience.md` |
| Master discovery `srun --nodes=1 -w "$MASTER_HOSTNAME" ip ...` | template comments |

## Conda Env Detection Rule

Working directory basename → env name:
- Lowercase the basename: `DiffSynth-valka` → `diffsynth-valka`
- **Trailing digits MUST be preserved**: `DiffSynth1` → `diffsynth1` (NOT `diffsynth`)

| Project dir basename | Correct env name | Wrong (don't use) |
|---|---|---|
| `DiffSynth-valka` | `diffsynth-valka` | `diffsynth_valka` |
| `DiffSynth1` | `diffsynth1` | `diffsynth` |
| `DiffSynth1-wan2-2-vae-lora` | `diffsynth1-wan2-2-vae-lora` | `diffsynth-wan2-2-vae-lora` |
| `LiveAvatar` | `liveavatar` | `live-avatar` |
| `MatAnyone` | `matanyone` | `matanyone1` |

If `ls ~/scratch/envs/` doesn't show your inferred
name, ask the user.

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
| Filling `TODO_account` with a non-`ehpc1003` value without being told to | Default to `ehpc1003`; user will say if the compute account has rotated. See `../../references/slurm.md` § 1 |
| `--gres=gpu:2` for "small test" | Always `--gres=gpu:4`; use `--nodes=1` for small tests |
| `acc_ehpc --time=00:30:00` for a quick test | Use `acc_debug` for tests; acc_ehpc only for real training |
| `conda activate diffsynth` for `DiffSynth1` dir | env name MUST match trailing digit: `diffsynth1` |
| Removing the NCCL block "for cleanliness" | All lines are BSC-validated; keep every one |
| Removing the `$TMPDIR` cache block | Required to avoid GPFS metadata storms and JIT lock contention |
| Auto-running `sbatch <new_script>` | NEVER. User reviews, user submits |

## Red Flags — STOP and reconsider

If you find yourself thinking any of these, STOP and re-read this skill:

- "I'll edit NCCL settings to optimize"
- "User asked for 2 GPUs, let me set --gres=gpu:2"
- "Daisy didn't tell me ehpc1003, but I see another account in `bsc_acct` — let me use that" (NO. Default to ehpc1003 unless explicitly told otherwise)
- "User said quick test → I'll use acc_ehpc with short --time"
- "I'll add `--mail-user=...`" (BSC SMTP is broken; we use Slack via job_notifier)
- "I'll skip the cache block to reduce script length"
- "User didn't give a dataset path — let me invent one based on the project name" (NO. ASK the user. Skill explicitly forbids guessing TODO_dataset_base / TODO_output_dir / TODO_ds_config.)
- "Delete `module load nccl/2.20.5` — conda env overrides it" (KEEP it as a defensive fallback; see slurm.md § 9)
- "DiffSynth1 dir → diffsynth env should work"
- "Let me sbatch it for the user, they're in a hurry"

All violations of invariants. Don't.

## What this skill does NOT do

- Inference scripts (inference runs directly on the compute node)
- Submitting the job (`sbatch` is always the user's call)
- Modifying the canonical template at `assets/slurm_template_multinode_train.sh`
- Data preprocessing scripts
- Benchmark or profiling scripts

## Background References

For the *why* behind any directive (read on demand, not eagerly):
- `../../references/hardware.md` — H100 / NVLink / NCCL fabric / IP-vs-RDMA
- `../../references/slurm.md` — QoS, accounts, switches, CPU rule, logs, SRUN
- `../../references/storage-layout.md` — projects/ vs scratch/, `${USER}`/`%u` portability rule
- `../../references/gpfs-hang-resilience.md` — D-state, probe pattern, why warmup matters
