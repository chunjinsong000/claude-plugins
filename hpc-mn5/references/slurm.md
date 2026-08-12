# BSC MareNostrum5 — SLURM Reference

QoS limits, account naming, partition selection, and walltime/switches
rules for the ACC partition. Read this when you need the *why* behind a
SBATCH directive in the SLURM template.

**Related**: `storage-layout.md` (paths) · `hardware.md` (H100/NCCL
fabric) · `gpfs-hang-resilience.md` (D-state, probe pattern).

---

## 1. Job-submission account vs storage account

BSC has **two distinct identifiers** that both look like `ehpcXXX` —
confusing them silently breaks job submission.

- **Storage account** = the GPFS allocation that owns
  `/gpfs/projects/...` and `/gpfs/scratch/...`. **See `storage-layout.md`
  § 1 for the canonical definition** (currently `ehpc1003`; verify the
  validity date with `bsc_acct` — see storage-layout.md § 1 note).
- **Job-submission account** = the compute allocation passed via
  `sbatch -A <id>` or `salloc -A <id>`. Tied to the project's GPU-hour
  budget; **rotates when GPU quota is exhausted**. The two accounts are
  usually different.

Run `bsc_acct` on a login node to list the job-submission accounts you
can use right now.

The SLURM template uses `--account=<account>` as a placeholder so the
script stays portable across compute-account rotations. The storage
paths are different — they hardcode the storage account name (see
`storage-layout.md`).

---

## 2. QoS Reference

Three QoSes are usable on the ACC partition. Only `acc_ehpc` and
`acc_debug` are relevant for training; `acc_interactive` is for
interactive exploration only.

| QoS | Priority | Max walltime | Concurrent running jobs | Max nodes/job | Submit-time limit | Use case |
|---|---|---|---|---|---|---|
| `acc_ehpc` | 100 | **3 days** (`3-00:00:00`) | many (per-user 366) | 100 | 366 | **Production training** |
| `acc_debug` | **10 000** (high) | **2 hours** | **1 at a time per user** | 8 | 366 | **Code debugging, fast turnaround** |
| `acc_interactive` | 100 | 2 hours | 1 | 1 | 366 | Interactive `srun --pty` for exploration |

### Selection rule

| You want to do | Use this QoS |
|---|---|
| Real training, ≥ 1 hour | `acc_ehpc` |
| 30 sec — 30 min smoke test | `acc_debug` |
| Type commands at a prompt on a GPU node | `acc_interactive` |

### Common mistakes
- ❌ Using `acc_ehpc` with `--time=00:30:00` for a quick test. **Wrong:**
  acc_debug has higher priority and will start sooner.
- ❌ Using `acc_debug` for a real 4-hour run. **Wrong:** 2-hour cap will
  truncate the job.
- ❌ Submitting two `acc_debug` jobs at once. SLURM will hold the second
  (since 2026-04-16: only 1 concurrent).

---

## 3. Walltime Rules

```
acc_debug      ≤ 02:00:00                    # 2 hours hard cap
acc_ehpc       ≤ 3-00:00:00                  # 3 days hard cap
acc_interactive ≤ 02:00:00                   # 2 hours hard cap
```

Always specify `--time=` explicitly. **Never** rely on queue defaults —
they're set conservatively low and will surprise you.

Format:
- `HH:MM:SS` for jobs ≤ 24h: `--time=02:00:00`
- `D-HH:MM:SS` for ≥ 1 day: `--time=2-12:00:00` (= 60 hours)

---

## 4. `--switches` Topology Hint

`--switches=count@max-wait` asks SLURM to schedule the job within at
most `count` IB leaf-switches, waiting up to `max-wait` for a tighter
allocation before falling back to scattered nodes.

| QoS | Recommended `--switches` | Why |
|---|---|---|
| `acc_debug` | `2@0:10:00` | 10-minute wait — short walltime, can't afford long backfill wait |
| `acc_ehpc` | `2@1:00:00` | 1-hour wait — production, worth waiting for tight topology to maximize IB performance |

Without `--switches`, SLURM may scatter your nodes across the cluster,
adding hops to NCCL inter-node traffic.

---

## 5. Partitions

| Partition | Nodes | What | When to use |
|---|---|---|---|
| `acc` | ~1120 | All GPU compute (4× H100 / node) | Almost everything: training, inference, benchmark |
| `accinteractive` | 1 (`alogin3`) | Interactive GPU shell | Only with `acc_interactive` QoS |
| `gpp` | many | GPP (CPU-only) login + compute | Pure CPU jobs (rare for us) |

The template hardcodes `acc` implicitly via `--gres=gpu:4` (only the `acc`
partition has GPUs). You don't need to write `--partition=acc`.

---

## 6. CPU and GPU Resource Rules

```
Per node:
  4 GPUs × 20 CPUs = 80 CPUs
```

| Directive | Value | Notes |
|---|---|---|
| `--gres=gpu:4` | full node | Don't request 1, 2, or 3 GPUs — partial-GPU is forbidden by BSC convention |
| `--cpus-per-task=80` | 4 GPUs × 20 | This is **per node** — multi-node submission keeps this at 80, not aggregated |
| `--ntasks-per-node=1` | one srun task | accelerate / torchrun spawns 4 sub-processes inside |
| `--nodes=N` | 1, 2, 4, 8, 16, 32, 64 | The only knob for scaling. Always whole nodes. |

**Why partial GPU is forbidden** — BSC's accounting and queue policy is
node-granular. Asking for 2 GPUs takes the whole node anyway, but burns
half the GPU-hour budget for nothing.

---

## 7. Log Output Directives — SLURM-side rules

The directory itself is defined in `storage-layout.md` § 6
(`/gpfs/scratch/ehpc1003/${USER}/slurm_logs/`). What's SLURM-specific:

- **Always use the `%`-format placeholders** so the script is portable
  across users:
  ```
  #SBATCH --output=/gpfs/scratch/ehpc1003/%u/slurm_logs/%x_%j.out
  #SBATCH --error=/gpfs/scratch/ehpc1003/%u/slurm_logs/%x_%j.err
  ```
  - `%u` → `$USER`, `%x` → job name, `%j` → job ID
  - Portability rule (no hardcoded usernames): `storage-layout.md` § 8
- **The directory must exist before submission.** SLURM does NOT
  auto-create it. The template handles this via `mkdir -p` early.
- **Never write logs under `$HOME`** — 80 GB quota fills fast with
  multi-MB stderr (NCCL chatter, traceback dumps).

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
  even though SLURM allocated 80 CPUs to the job. Throughput tanks
  silently — no error.
- Setting this env var makes `srun` re-read SLURM's allocation and
  propagate to its child processes correctly.

---

## 9. `module load nccl/2.20.5` — kept as fallback

The template loads `nccl/2.20.5` even though every modern conda env
(PyTorch ≥ 2.x) ships its own bundled NCCL in
`$CONDA_PREFIX/lib/libnccl.so.2.X`. At runtime the conda env wins via
`LD_LIBRARY_PATH`, so the system module is **silently shadowed** in the
common case.

We keep the load anyway as a **defensive fallback** for two scenarios:
- If the conda env is missing or `LD_LIBRARY_PATH` is wrong (e.g. user
  forgot `conda activate` or activated a CPU-only env), the system NCCL
  is still on the dynamic linker path. Bare `import torch` won't crash
  on missing `libnccl`.
- If a future env strips NCCL (e.g. CPU-only build), inter-node
  collectives still resolve to a working binary.

Net cost: ~50 ms at job start to add the module's lib path. Worth keeping.

---

## 10. Submit-Time Sanity Checks

Before `sbatch`, verify:

```bash
# 1. Is your job-submission account current?
bsc_acct

# 2. Is your storage account still valid? (storage-layout.md § 1 — ehpc1003)
ls ~/scratch/

# 3. Will the script syntax-check?
bash -n train/<job_name>.sh

# 4. Did all TODO_* placeholders get filled?
grep -c TODO_ train/<job_name>.sh   # MUST be 0

# 5. Does your conda env exist where the script expects?
ls ~/scratch/envs/<env_name>
```

---

## 11. Common SLURM Commands

| What you want | Command |
|---|---|
| List your accounts | `bsc_acct` |
| Show your queued/running jobs | `squeue -u $USER` |
| Predicted start time | `squeue -j <jobid> --start` |
| Why is it pending | `squeue -j <jobid> -o "%R"` |
| Past job stats | `sacct -j <jobid> -o JobID,JobName,State,ExitCode,Elapsed,NodeList` |
| Cancel a job | `scancel <jobid>` |
| Your QoS limits | `sacctmgr show qos acc_ehpc format=Name,MaxWall,MaxJobsPU,MaxSubmitPU,MaxTRESPU` |
| Your fairshare | `sshare -u $USER --format=Account,User,FairShare,RawShares,EffectvUsage` |

---

## 12. Failure Modes & Quick Diagnoses

| Symptom | Likely cause | Fix |
|---|---|---|
| `sbatch: error: invalid account` | account placeholder not filled | grep for `<account>` in script, replace with `bsc_acct` output |
| `sbatch: error: Invalid qos specification` | QoS doesn't allow that account | check `sacctmgr show user $USER format=user,account,defaultaccount,qos` |
| `Job ... PartitionTimeLimit` | walltime > QoS max | reduce `--time=` or switch QoS |
| `Job ... AssocGrpJobsLimit` | already running too many of this QoS | wait, or cancel a running one (1-job limit on acc_debug) |
| Job runs forever in PENDING | no nodes match topology constraint | check `--switches=` value; relax to longer wait or remove |
| `srun: error: ... Unable to create job step` | requesting more CPUs than allocated | check `SRUN_CPUS_PER_TASK` is set |

---

## 13. References

- BSC user docs: <https://www.bsc.es/supportkc/docs/MareNostrum5/intro/>
- This skill's hardware reference: `references/hardware.md`
- BSC support tickets that informed the NCCL config in the template:
  - **#428699** — Jon Navarro, NCCL recommendations, 2026-04-16
- Companion onboarding repo: `~/projects/temp/bsc-onboarding/docs/`
