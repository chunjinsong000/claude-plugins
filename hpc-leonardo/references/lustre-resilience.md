# LEONARDO Lustre — Hang Resilience, Probes & Stripe Configuration

How to defend long-running scripts against Lustre metadata stalls, recover from client evictions, and tune large-file I/O via stripe configuration. Read when writing any code that does file I/O over `/leonardo_work/` or `/leonardo_scratch/` — which is nearly all training code.

**Related**: [`storage-layout.md`](../../../../docs/storage-layout.md) (which paths exist) · `slurm.md` § "SLURM `--kill-on-bad-exit` semantics".

---

## 1. Lustre primer (what's underneath `/leonardo*`)

LEONARDO's parallel filesystem is **Lustre 2.14.0_ddn182** (DDN-branded fork) over InfiniBand (`@o2ib` LNET). Three mounts on every node:

| Mount | What it is | OSTs / MDTs |
|---|---|---|
| `/leonardo` (= `$HOME`) | Small home areas, dotfiles only | 8 MDTs, 6 OSTs (~85 TB total) |
| `/leonardo_scratch` | Per-user scratch, 40-day purge | 8 MDTs, 6 OSTs (~3 PB total) |
| `/leonardo_work` | Project-shared work area | 8 MDTs, 6 OSTs (~3 PB total) |

Components:

| Term | Role | Failure impact |
|---|---|---|
| **MGS** (Management Server) | Cluster config — small, rarely hot | If down, no new clients can mount |
| **MDS / MDT** (Metadata Server / Target) | `stat`, `open`, `mkdir`, `readdir`, locks on metadata | If one MDT slow: directory listings stall |
| **OSS / OST** (Object Storage Server / Target) | File data reads/writes; locks on byte ranges | If one OST slow: any file *striped on it* stalls |
| **LNET** (Lustre Networking, here over IB) | RPC transport | If saturated: client timeouts → eviction |
| **LDLM** (Lustre Distributed Lock Manager) | Token issuing — IBITS locks (metadata) + extent locks (data) | Lock contention → spurious slow I/O |

Each file is **striped** across 1 to all OSTs (default = 1). When a client opens a file, it gets extent locks from the OSTs holding its stripes; those locks live in the LDLM namespace until released or revoked.

## 2. What is a Lustre "hang"?

Three distinct symptoms get called "Lustre is stuck", and they require different responses:

### 2a. D-state hang (kernel-blocking syscall)

`stat`/`open`/`readdir` waits for the server to reply and the process is in **D state** ("uninterruptible sleep") on the kernel side:

```bash
$ ps aux | grep ls
$USER... D ... 0:00 ls /leonardo_scratch/.../some-dir/
```

| Signal | Effect |
|---|---|
| `SIGTERM` (`kill`) | ❌ ignored — kernel-mode |
| `SIGKILL` (`kill -9`) | ❌ ignored — kernel-mode |
| `Ctrl+C` | ❌ ignored |
| `scancel <jobid>` | ❌ for D-state ranks |

Lustre's client-side timeout (default ~100s) eventually triggers either **success (server eventually replied)** or **eviction** (see 2b).

### 2b. Client eviction (Lustre-specific)

If a client doesn't send keep-alive RPCs to the server, or the server thinks the client is misbehaving, the **server evicts the client**. New I/O on that mount returns **`-ESTALE`** or **`-EIO`**:

```
$ cat /leonardo_work/AIFAC_F02_378/file
cat: /leonardo_work/AIFAC_F02_378/file: Stale file handle
```

The kernel logs:

```bash
dmesg | tail -20
# ... Lustre: ... ost15: This client was evicted by ost15; in progress operations using this service will fail.
```

The session usually self-heals after 30-60 s (client re-mounts, locks re-acquired) but **anything in flight at eviction time is lost**. Distinct from D-state in that **errors return immediately**; your code sees an `OSError` rather than a hang.

### 2c. LDLM lock contention

Many clients hammering the same path → LDLM serializes via locks. Symptoms:
- `stat` returns in 50 ms normally but 5–30 s under load
- High `lock_count` on a specific MDT or OST (see § 4)
- Doesn't trigger D-state, just gets slow

### When each happens

| Trigger | Frequency | Symptom |
|---|---|---|
| Multi-node cold start (`import torch` on 16 nodes at once) | every multi-node start | D-state on cold inodes |
| Bulk small-file I/O on one MDT (millions of files) | when someone does it | 2c (lock contention) |
| Filesystem maintenance / OST failover | rare (CINECA announces in HPC News) | 2a + 2b combined |
| Single client misbehaving (we sent too many ops) | self-inflicted | 2b (eviction) |
| Single OST overloaded by another user's job | unpredictable | 2c, possibly 2a |
| 20:00–04:00 CEST nightly batch peaks | weekly | mixed |

---

## 3. The probe philosophy

You **cannot prevent** a process from blocking once it's already in a syscall — kernel state, no userspace rescue.

You **can avoid** initiating expensive I/O when Lustre is already known-unhealthy:

> Before doing N seconds of unrecoverable I/O, do 1–10 seconds of probing. If the probe fails, fail fast (exit non-zero) instead of hanging.

The probe is a minimal syscall (`stat`) wrapped in `timeout`, plus optionally a `lfs check servers` snapshot if you want a quick "is the whole fs healthy" view.

### Probe = `stat` + `timeout`

```bash
PROBE_TIMEOUT=10  # seconds

probe_lustre() {
    local path="$1"
    if ! timeout "$PROBE_TIMEOUT" stat "$path" > /dev/null 2>&1; then
        echo "[$(date +%T)] $(hostname) LUSTRE PROBE FAILED — stat '$path' > ${PROBE_TIMEOUT}s" >&2
        return 1
    fi
}

# Usage at start of any expensive op
probe_lustre "/leonardo_work/AIFAC_F02_378/${USER}/avatar/code" || exit 1
probe_lustre "/leonardo_scratch/large/userexternal/${USER}/avatar/checkpoints" || exit 1
# ... now do the actual I/O ...
```

This is **identical to the BSC probe pattern** — kernel D-state is the same on any network filesystem. Both `lfs` and `stat` go through the same kernel path.

### Probe = `lfs check servers` (cluster-wide health)

For a quick "are all OSTs reachable from this node":

```bash
lfs check servers 2>&1 | grep -v active. | head    # any non-"active" lines = problem
```

Healthy looks like:
```
larchive-OST0000-osc-ff486680c4e89000 active.
larchive-OST0001-osc-ff486680c4e89000 active.
...
```

A failing OST shows `inactive`, `failover`, or a timeout — bail out before starting your I/O.

Optionally combine:

```bash
probe_lustre_full() {
    local path="$1"
    # 1) Quick path probe
    timeout 10 stat "$path" > /dev/null 2>&1 || { echo "stat failed"; return 1; }
    # 2) Cluster-wide health
    local bad=$(lfs check servers 2>&1 | grep -v 'active.' | head -3)
    if [ -n "$bad" ]; then
        echo "LUSTRE OSTs/MDTs not all active:"; echo "$bad"
        return 1
    fi
}
```

### What probe does NOT do

- ❌ Save you from a hang already in progress.
- ❌ Make Lustre faster.
- ❌ Detect every stall (probe at T=0 says nothing about T+5 s).
- ❌ Detect a bad stripe layout — that's `lfs getstripe`, see § 6.

### What probe DOES do

- ✅ Catch "Lustre is currently bad" before a 30-min copy.
- ✅ Let multi-node SLURM jobs bail at job start (via `--kill-on-bad-exit=1`) instead of dying 900 s later in NCCL TCPStore rendezvous.
- ✅ Surface clear diagnostics for `.err` log grep.

---

## 4. LDLM lock contention — diagnose, don't probe

LDLM-induced slowness (2c) doesn't fail a probe — it just adds 5–30 s per op. To diagnose live:

```bash
# Lock count per Lustre namespace (per MDT and per OST connection)
lctl get_param -n ldlm.namespaces.*.lock_count | sort -nk3 | tail -10

# Drill into a specific busy namespace
lctl get_param ldlm.namespaces.larchive-MDT0001-mdc-*.contention_seconds
lctl get_param ldlm.namespaces.larchive-MDT0001-mdc-*.contended_locks
```

| Reading | What it means |
|---|---|
| `lock_count > 10000` on one MDT | That MDT is hot — your dir or someone else's is causing it |
| `contention_seconds > 1.0` average | Real contention; you're waiting for tokens |
| Many namespaces all around `0–100` | Healthy |

If your own job is the cause, the fix is usually **don't open the same file from 16 ranks at once** — fan out via a directory hash, or have rank 0 read and broadcast.

---

## 5. Eviction recovery

If your code sees `ESTALE` / `EIO` mid-job:

```bash
# 1) Confirm via kernel log
dmesg | tail -20 | grep -i lustre
# Look for: "This client was evicted by ostN"

# 2) The mount usually self-heals in 30-60 s. Retry the operation:
sleep 30
stat /leonardo_work/AIFAC_F02_378/  # should succeed now

# 3) If it doesn't heal in 5 min, report to superc@cineca.it with:
#    - hostname (uname -n)
#    - timestamp of eviction (from dmesg)
#    - what you were doing (rsync? training? open?)
```

In training code, the right thing is to **catch `OSError` around critical writes** (checkpoint save, config dump), retry with backoff, and emit a clear log line so post-mortem analysis sees it:

```python
import time, errno

def robust_save(save_fn, path, attempts=3, backoff=30.0):
    for attempt in range(1, attempts + 1):
        try:
            save_fn(path)
            return
        except OSError as e:
            if e.errno in (errno.ESTALE, errno.EIO):
                print(f"[lustre] save attempt {attempt} got errno {e.errno}; "
                      f"sleeping {backoff}s before retry", flush=True)
                time.sleep(backoff)
                backoff *= 2
            else:
                raise
    raise RuntimeError(f"Lustre eviction-retry exhausted on {path}")
```

---

## 6. Stripe configuration — Lustre's killer feature you must learn

**Lustre makes you decide** how many OSTs each file is striped across. Default is **1 OST per file**, which means:

- Single-OST bandwidth ≈ 500 MB/s–2 GB/s
- 4× stripe ≈ 4× bandwidth (close to linear for sequential I/O)

For datasets and large checkpoints, **default stripe is wrong**.

### Inspect & set stripe

```bash
# What stripe does this file/dir currently have?
lfs getstripe path/to/file
# Look for "stripe_count: N", "stripe_size: 1048576" (1 MiB)

# Set stripe BEFORE creating files (stripe is fixed at file creation):
mkdir -p /leonardo_work/AIFAC_F02_378/${USER}/avatar/datasets
lfs setstripe -c 4 -S 4m /leonardo_work/AIFAC_F02_378/${USER}/avatar/datasets

# Files created INSIDE that dir inherit the 4-OST, 4 MiB stripe.

# To re-stripe existing files:
lfs migrate -c 8 /leonardo_work/AIFAC_F02_378/.../big.tar
```

### Recommended stripe by file type

| File pattern | Recommended stripe | Why |
|---|---|---|
| Small text / configs / `.py` / Python wheels | `-c 1` (default) | Latency-bound; striping adds overhead |
| Conda envs (`$WORK/$USER/envs/*`) — many small files | `-c 1` | Small-file I/O; striping pessimizes |
| Single-file model checkpoints (1–100 GB) | `-c 4` to `-c 8` | Sequential read/write at full bandwidth |
| Training datasets (HF, large parquet / video files) | `-c 4` on the directory | Multiple ranks read in parallel |
| `.sif` Singularity images (5–30 GB) | `-c 4` | Loaded once at start, big sequential read |
| TensorBoard / wandb event logs | `-c 1` | Tiny appends, latency-bound |

### Set defaults for your project dirs

Run once after `$WORK` is provisioned:

```bash
# Conda envs and slurm logs: leave default (small files)
# Datasets and checkpoints: 4-OST stripe
lfs setstripe -c 4 -S 4m $WORK/shared/datasets/
lfs setstripe -c 4 -S 4m $WORK/shared/models/
lfs setstripe -c 4 -S 4m $WORK/shared/singularity/
lfs setstripe -c 4 -S 4m $SCRATCH/checkpoints/  # if you mkdir this ahead
```

The `-S 4m` sets the stripe size (chunk that goes round-robin between OSTs). 4 MiB is a good default for sequential reads of big files. For small-file workloads, leave it at default (1 MiB).

### Verify stripe on a real file

```bash
$ lfs getstripe my-checkpoint.pt
my-checkpoint.pt
lmm_stripe_count:  4
lmm_stripe_size:   4194304
lmm_pattern:       raid0
...
```

A stripe miss is silent — files end up on 1 OST, training-time I/O is 4× slower than it could be, no error message. **`lfs getstripe` early, `lfs getstripe` often.**

---

## 7. When to probe / when to set stripe

| Situation | Probe? | Stripe? |
|---|---|---|
| Long-running training script, before each checkpoint save | ✅ defensive | ✅ checkpoint dir |
| `mkdir -p` at start of job | ✅ probe parent | ✅ set stripe on dir before first write |
| `rsync` / `cp` of large dataset | ✅ before each chunk | ✅ destination dir |
| Multi-node SLURM job startup (warmup) | ✅ in `srun --kill-on-bad-exit=1` | once at dataset-build time |
| Reading a config file (small, one-shot) | ❌ overkill | ❌ default |
| Inside a tight loop reading 10000 files | ❌ probe at chunk boundaries | ✅ on the dir |
| Already in a hang | ❌ too late — see § 5 / wait it out | n/a |

### Pattern: probe-then-IO at chunk boundaries

```bash
chunk=0
for file in "${huge_list[@]}"; do
    if (( chunk % 100 == 0 )); then
        probe_lustre "$SCRATCH/${USER}/X/" || {
            echo "Lustre sick at chunk $chunk; sleeping 60s and retrying once..."
            sleep 60
            probe_lustre "$SCRATCH/${USER}/X/" || exit 1
        }
    fi
    cp "$file" $SCRATCH/...
    ((chunk++))
done
```

---

## 8. SLURM-specific patterns

### Bail-fast at job startup

```bash
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash $WORK/$USER/scripts/lustre-probe.sh "$CONDA_PREFIX" "$SCRATCH/$PROJECT"
```

`--kill-on-bad-exit=1` — if **any** node's `srun` step returns non-zero, SLURM kills the rest. Combined with the probe inside `lustre-probe.sh` (which exits 1 on probe fail), this means a single sick node aborts the entire multi-node job at startup, instead of waiting 900 s in NCCL TCPStore rendezvous.

### Cost analysis

Without probe + `--kill-on-bad-exit`:
```
[T+0s]   Job starts on 16 nodes; 1 node has stuck Lustre
[T+5s]   15 nodes import torch successfully, hit NCCL barrier
[T+5s]   1 sick node: import torch hangs (D-state on libtorch.so stat)
[T+905s] NCCL TCPStore times out
[T+905s] All ranks crash with "DistStoreError"
        Cost: 16 × 905 s × 4 GPU ≈ 16 GPU-h wasted
```

With probe + `--kill-on-bad-exit`:
```
[T+0s]   Job starts on 16 nodes
[T+10s]  lustre-probe.sh probes — sick node fails
[T+11s]  --kill-on-bad-exit=1 cancels all other nodes
[T+12s]  Job ends as FAILED
        Cost: 16 × 12 s × 4 GPU ≈ 0.21 GPU-h
```

**~75× cheaper failure**, just like BSC.

---

## 9. What to do when Lustre is actually hung

If you're staring at a hung `cd` / `ls` / `mkdir` right now:

1. **Don't `kill -9`** — won't work, wastes a minute.
2. **Open a new shell** (the old one is held hostage by the syscall).
3. **Check `dmesg | tail`** — Lustre logs evictions, lock contention, OST timeouts there.
4. **Check `lfs check servers`** in the new shell — if some OSTs aren't `active.`, it's a cluster-wide problem.
5. **Wait 30–120 s** — most stalls self-resolve when the server catches up or eviction completes.
6. **Don't submit new jobs during a hang** — SLURM's prolog will also hang on the same fs.
7. **If still hung after 5 min**: check CINECA HPC News (https://www.hpc.cineca.it/hpc-center-news/) for maintenance / incident announcements. If clear, email `superc@cineca.it` with `hostname`, timestamps from `dmesg`, what you were doing.

### What you cannot do

- **Force-unmount and remount Lustre** — root-only.
- **Terminate the D-state process** — see § 2a; untrappable.
- **Recover unwritten data** — anything that didn't `fsync` before the hang is gone.

---

## 10. References

- LEONARDO's Lustre version: `lfs 2.14.0_ddn182` (DDN fork, the hardware vendor for `/leonardo_*` storage)
- CINECA HPC News (for cluster-wide incidents): https://www.hpc.cineca.it/hpc-center-news/
- Storage layout & path conventions: [`docs/storage-layout.md`](../../../../docs/storage-layout.md)
