# LEONARDO Booster — SLURM Reference

QoS limits, account naming, partition selection, walltime rules, and
log-output conventions for the `boost_usr_prod` partition. Read this
when you need the *why* behind a SBATCH directive in the SLURM template.

**Related**: `storage-layout.md` (paths) · `hardware.md` (A100 / NCCL
fabric on Booster) · `lustre-resilience.md` (Lustre stalls, eviction).

> Authoritative QoS table lives in the parent `SKILL.md` § QoS Selection.
> This file expands on the *why* and adds command snippets; never restate
> the cap numbers here, point back.

---

## 1. Account / partition / QoS — one allocation, several QoS

Leonardo has a single compute account for the team: **`AIFAC_F02_378`**
(case-insensitive; `sacctmgr` reports it lowercase as `aifac_f02_378`).
The partition is **`boost_usr_prod`** and it has four QoS — see the
table in `SKILL.md` § QoS Selection.

Storage account naming follows the same string (`AIFAC_F02_378`); the
allocation is valid until **2027-03-25**.

There is no separate "job-submission account" that rotates — unlike some
BSC-style allocations. The same string is used for `--account=` and for
the `$WORK` path component (`/leonardo_work/AIFAC_F02_378/`).

```bash
sacctmgr -p show assoc where user=$USER format=Account,QOS,Partition
# Account|QOS|Partition|Def QOS|
# aifac_f02_378|boost_qos_bprod,boost_qos_dbg,boost_qos_lprod,normal|||
```

---

## 2. QoS — see § QoS Selection in SKILL.md

`SKILL.md` is the single source of truth for the four QoS:
- `boost_qos_dbg` — debug
- `normal` — standard production (≤ 24 h)
- `boost_qos_bprod` — big production (many nodes, ≤ 24 h)
- `boost_qos_lprod` — long production (≤ 8 nodes, ≤ 4 days)

This file does NOT restate caps. To query live limits:
```bash
sacctmgr -p show qos format=Name,MaxWall,MaxNodes,MaxTRESPerJob \
         name=boost_qos_dbg,boost_qos_bprod,boost_qos_lprod,normal
```

### Common QoS-selection mistakes

- ❌ Using `normal` with `--time=00:30:00` for a quick test → `boost_qos_dbg`
  has higher priority and starts sooner.
- ❌ Using `boost_qos_dbg` for a 1-hour run → 30-minute cap will truncate.
- ❌ Using `normal` for a 48-hour run → exceeds 24 h cap; either chunk
  into checkpoints or switch to `boost_qos_lprod`.
- ❌ Using `boost_qos_lprod` to scale to 16 nodes → 8-node cap; switch to
  `normal` / `boost_qos_bprod` and split walltime.

---

## 3. Walltime format

Always specify `--time=` explicitly. Format:
- `HH:MM:SS` for jobs ≤ 24 h: `--time=02:00:00`
- `D-HH:MM:SS` for ≥ 1 day: `--time=2-12:00:00` (= 60 hours)

Hard caps per QoS: see `SKILL.md` § QoS Selection.

---

## 4. `--switches` topology hint (Leonardo: optional)

`--switches=count@max-wait` asks SLURM to schedule the job within at
most `count` IB leaf-switches, waiting up to `max-wait` for a tighter
allocation. On Leonardo the IB fabric is fast enough that this rarely
helps for ≤ 8-node jobs, and may delay scheduling.

| When to add `--switches` | When to skip |
|---|---|
| ≥ 16-node training, latency-sensitive comms (e.g. ZeRO-3, sequence parallel) | ≤ 8-node jobs, debug runs, anything where queue priority > rail tightness |

The templates **do not include** `--switches` by default. Add it only if
profiling shows inter-node comm is the bottleneck and you have spare
queue patience.

---

## 5. Partition: only `boost_usr_prod` for GPU work

Other partitions exist (`dcgp_usr_prod` is CPU-only, `lrd_all_viz` is
visualization, etc.) but the SLURM template assumes
`--partition=boost_usr_prod` because that's the only one with `gpu:a100:4`.

```bash
sinfo -o "%P %l %D %c %m %G" 2>&1 | head -5
# boost_usr_prod   1-00:00:00  3196  32  514000  gpu:a100:4    ← this one
```

---

## 6. CPU and GPU resource rules — Leonardo Booster

```
Per node:
  4 × A100-SXM-64GB GPUs
  32 logical CPUs (Intel Xeon Platinum 8358 Ice Lake, 2 sockets × 16 cores)
  514 GB RAM
  10 GB tmpfs at /tmp
```

| Directive | Value | Notes |
|---|---|---|
| `--gres=gpu:4` | full node | Site convention: GPU is whole-node only |
| `--cpus-per-task=32` | Leonardo Booster CPU count | 32 logical CPUs / 4 GPUs = 8 per rank — propagated via `SRUN_CPUS_PER_TASK` |
| `--ntasks-per-node=1` | one srun task | accelerate / torchrun spawns 4 sub-processes inside |
| `--nodes=N` | 1, 2, 4, 8, 16, … | Whole nodes only |

Why partial GPUs are not allowed: Leonardo accounts and the
`boost_usr_prod` partition are node-granular for billing; requesting 2
GPUs occupies the whole node but burns half the GPU-hours for nothing.

---

## 7. Log output directives — Leonardo paths

The shared SLURM log directory is **`/leonardo_work/AIFAC_F02_378/$USER/slurm_logs/`**
(under `$WORK` so it survives `$SCRATCH` purges).

- **SBATCH directives must use absolute paths** (no shell-variable
  expansion). Use SLURM placeholders for the user-portable parts:
  ```
  #SBATCH --output=/leonardo_work/AIFAC_F02_378/%u/slurm_logs/%x_%j.out
  #SBATCH --error=/leonardo_work/AIFAC_F02_378/%u/slurm_logs/%x_%j.err
  ```
  - `%u` → `$USER`, `%x` → job name, `%j` → job ID
- **The directory must exist before submission.** SLURM does NOT
  auto-create. One-time setup:
  ```bash
  mkdir -p /leonardo_work/AIFAC_F02_378/$USER/slurm_logs
  ```
- **Never write logs under `$HOME`** — Leonardo home has a 50 GB quota
  that NCCL chatter / stack dumps will fill fast.

---

## 8. `SRUN_CPUS_PER_TASK` (required since SLURM 22.05+)

```bash
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
```

Why this is mandatory:

- Pre-22.05 SLURM, `srun` inside a batch script automatically inherited
  `--cpus-per-task` from the SBATCH directive.
- Since 22.05, that auto-inheritance was removed. `srun` defaults to **1
  CPU per task** unless explicitly told otherwise.
- Without this export, your DataLoader workers all collide on one core,
  even though SLURM allocated 32 CPUs to the job. Throughput tanks
  silently — no error.
- Setting this env var makes `srun` re-read SLURM's allocation and
  propagate `cpus-per-task` to its child processes correctly.

---

## 9. Email notifications

The templates ship with:
```
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alex.liu@valka.ai
```

Leonardo's mail relay works. `END` + `FAIL` is the recommended pair —
`BEGIN` is too noisy for production runs that queue for hours. Change
the address per user if the template is reused by a teammate.

---

## 10. Submit-time sanity checks

Before `sbatch`, verify:

```bash
# 1. Account / QoS / partition are valid for you?
sacctmgr -p show assoc where user=$USER format=Account,QOS,Partition

# 2. Storage allocation still valid? (Until 2027-03-25 for AIFAC_F02_378)
ls $WORK

# 3. Script syntax checks?
bash -n train/<job_name>.sh

# 4. All TODO_* placeholders filled?
grep -c TODO_ train/<job_name>.sh   # MUST be 0

# 5. Conda env exists?
ls /leonardo_work/AIFAC_F02_378/$USER/conda/envs/<env_name>

# 6. Log dir exists?
ls -d /leonardo_work/AIFAC_F02_378/$USER/slurm_logs
```

---

## 11. Common SLURM commands

| What you want | Command |
|---|---|
| Show your accounts/QoSes | `sacctmgr -p show assoc where user=$USER format=Account,QOS,Partition` |
| Show your queued/running jobs | `squeue -u $USER` |
| Predicted start time | `squeue -j <jobid> --start` |
| Why is it pending | `squeue -j <jobid> -o "%R"` |
| Past job stats | `sacct -j <jobid> -o JobID,JobName,State,ExitCode,Elapsed,NodeList` |
| Cancel a job | `scancel <jobid>` |
| QoS limits | `sacctmgr show qos <qos-name> format=Name,MaxWall,MaxNodes,MaxTRESPerJob` |
| Your fairshare | `sshare -u $USER --format=Account,User,FairShare,RawShares,EffectvUsage` |

---

## 12. Failure modes & quick diagnoses

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: invalid account` | account placeholder not filled | `grep TODO_account` in script; replace with `AIFAC_F02_378` |
| `sbatch: error: Invalid qos specification` | QoS not allowed for this account | check `sacctmgr -p show assoc where user=$USER format=Account,QOS` |
| `Job ... PartitionTimeLimit` | walltime > QoS max | reduce `--time=` or switch QoS (see § QoS Selection) |
| `Job ... AssocGrpJobsLimit` | already running too many of this QoS | wait, or scancel a running one |
| Job runs forever in PENDING | low priority or no matching nodes | check `squeue -j <id> -o "%R"` (Reason field) |
| `srun: error: ... Unable to create job step` | requesting more CPUs than allocated | confirm `SRUN_CPUS_PER_TASK` is exported |
| `Failed to initialize NVML: Driver/library version mismatch` | LD_LIBRARY_PATH put a wrong libnvidia-ml.so first | drop `$CUDA_HOME/lib64` from LD_LIBRARY_PATH; `CUDA_HOME + PATH` only (see SKILL.md Red Flags) |

---

## 13. References

- Leonardo user docs: <https://docs.hpc.cineca.it/hpc/leonardo.html>
- Hardware specifics: `references/hardware.md`
- Storage paths: `references/storage-layout.md`
- Lustre health: `references/lustre-resilience.md`
- NCCL recommendations sourced from BSC support ticket **#428699**
  (Jon Navarro, 2026-04-16) — same IB-tuning principles apply on
  Leonardo Booster with adapted HCA names; see `hardware.md` § NCCL.
