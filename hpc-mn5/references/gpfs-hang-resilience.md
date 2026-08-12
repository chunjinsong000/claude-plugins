# BSC GPFS — Hang Resilience & Probe Pattern

How to defend long-running scripts against GPFS metadata stalls. Read
when writing any code that does file IO over `/gpfs/...` (which is
nearly all training code).

**Related**: `storage-layout.md` (which paths exist) · `slurm.md` § 6
(SLURM `--kill-on-bad-exit` semantics).

---

## 1. What is a GPFS hang?

`/gpfs/...` is a parallel network filesystem shared across thousands of
BSC nodes. When the metadata server is slow, the IB fabric to the
storage nodes flaps, or someone is doing bulk small-file IO, your basic
syscalls (`stat`, `open`, `readdir`) will **block in the kernel until
the IO returns** — sometimes minutes, sometimes hours.

The processes show up as **D state** ("uninterruptible sleep") in `ps`:

```bash
$ ps aux | grep ls
vlk... D ... 0:00 ls /gpfs/scratch/.../some-dir/
```

Once a process is in D state on a stuck GPFS syscall:

| Signal | Effect |
|---|---|
| `SIGTERM` (`kill`) | ❌ ignored — process is in kernel-mode |
| `SIGKILL` (`kill -9`) | ❌ ignored — process is in kernel-mode |
| `Ctrl+C` | ❌ ignored |
| `scancel <jobid>` | ❌ for D-state ranks (job slot held until kernel returns) |

Only the kernel can interrupt it (eventually, when the syscall returns
or times out internally). For your script, **the rank is dead until then**.

### When it happens

| Trigger | How often |
|---|---|
| 3-4 AM Spain time (other users' bulk transfer windows) | weekly |
| Many ranks doing first-time `import torch` simultaneously (cold metadata) | every multi-node cold start |
| Concurrent jobs hammering the same dir | unpredictable |
| Filesystem maintenance / metadata-server failover | rarely (BSC announces) |

---

## 2. The probe philosophy

You **cannot prevent** a process from hanging once it's already in a
syscall — that's a kernel state, no userspace code can rescue it.

You **can avoid** initiating expensive IO when GPFS is already
known-unhealthy. The pattern:

> **Before doing N seconds of unrecoverable IO, do 1 second of probing.
> If the probe fails, fail fast (exit non-zero) instead of hanging.**

The probe is a single `stat` (or similar minimal syscall) wrapped in a
`timeout`. If the probe takes longer than the deadline, you assume GPFS
is hung and bail out *before* the long IO starts.

### What probe DOES NOT do

- ❌ Save you from a hang that has already started.
- ❌ Make GPFS faster.
- ❌ Detect every stall (a probe at T=0 says nothing about T+5s).
- ❌ Replace good error handling.

### What probe DOES

- ✅ Catch "GPFS is currently bad" before you start a 30-min copy.
- ✅ Let multi-node SLURM jobs **bail at job start** (via
  `--kill-on-bad-exit=1`) instead of waiting 900s in NCCL rendezvous on
  a stuck node.
- ✅ Surface clear diagnostics (`PROBE FAILED on node X path Y in 10s`)
  so you can grep `.err` files later.

---

## 3. The canonical probe (Bash)

```bash
PROBE_TIMEOUT=10  # seconds — short enough to not waste, long enough to allow load spikes

probe_gpfs() {
    local path="$1"
    if ! timeout "$PROBE_TIMEOUT" stat "$path" > /dev/null 2>&1; then
        echo "[$(date +%T)] $(hostname) GPFS PROBE FAILED — stat '$path' > ${PROBE_TIMEOUT}s" >&2
        return 1
    fi
}

# Usage: probe before any expensive operation
probe_gpfs "/gpfs/scratch/ehpc1003/${USER}/checkpoints" || exit 1
# ... now do the actual IO ...
```

**Key choices:**

- `timeout <N>` from coreutils — sends SIGTERM after N seconds. The `stat`
  itself may not respond (it's in D-state), but `timeout` will eventually
  give up and return rc=124. Your script then continues.
- `stat` (not `ls`) — `ls` reads the dir, `stat` only reads the inode.
  Cheaper, less likely to false-positive.
- Two depths recommended — probe both the mount root and the actual
  target subdir. A mount-level stall and an inner-tree stall have
  different causes.

### In-house reference implementation

`$HOME/scripts/warmup_page_cache.sh` is our production probe + warmup:

```bash
# ---- GPFS health probe (fail fast if FS is hung) ----
for path in "$ENV_PATH" "$SP"; do
    if ! timeout "$PROBE_TIMEOUT" stat "$path" > /dev/null 2>&1; then
        echo "[$(date +%T)] $HOST GPFS PROBE FAILED — stat '$path' > ${PROBE_TIMEOUT}s" >&2
        exit 1
    fi
done
```

The SLURM template invokes it inside `srun --kill-on-bad-exit=1` so a
single bad node aborts the whole multi-node job at startup, instead of
limping along until rendezvous timeout.

---

## 4. The canonical probe (Python)

```python
import subprocess, sys, os

def probe_gpfs(path: str, timeout: float = 10.0) -> bool:
    """Probe a GPFS path with a hard timeout.

    Returns True if `stat` returns within `timeout` seconds, False otherwise.
    Never raises — D-state subprocesses are reaped by `timeout`'s kill.
    """
    try:
        subprocess.run(
            ["stat", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
        )
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return False


# Usage at the top of long-running code paths
ckpt_dir = os.path.expanduser("~/scratch/MyProject/ckpts")
if not probe_gpfs(ckpt_dir):
    print(f"GPFS unhealthy: {ckpt_dir} not reachable in 10s; aborting.", file=sys.stderr)
    sys.exit(1)

# ... now save checkpoint ...
```

**Why `subprocess.run` and not `os.stat`:**

- `os.stat()` from Python is a single syscall and **will block in the
  same way as the kernel** — Python's `signal.alarm()` cannot interrupt
  a syscall in D-state, only between syscalls.
- `subprocess.run(..., timeout=N)` spawns a child that does the syscall.
  When `timeout` expires, the parent kills the child. Even though the
  child stays D-state, the parent code resumes immediately because the
  child is no longer a foreground process for our flow.

---

## 5. When to probe

| Situation | Probe? |
|---|---|
| Long-running training script, before each checkpoint save | ✅ optional, defensive |
| `mkdir -p` at the start of a job | ✅ probe parent first |
| `rsync` / `cp` of large dataset | ✅ probe before each chunk |
| Multi-node SLURM job startup (warmup) | ✅ already done by `warmup_page_cache.sh` |
| Reading a config file (small, one-shot) | ❌ overkill |
| Inside a tight loop reading 10000 files | ❌ would amortize; probe at chunk boundaries |
| Already in a hang (you're reading this in despair) | ❌ too late — wait it out or `scancel` (best-effort) |

### Pattern: probe-then-IO at chunk boundaries

For very long IO loops (hours), probe periodically:

```bash
chunk=0
for file in "${huge_list[@]}"; do
    if (( chunk % 100 == 0 )); then
        probe_gpfs "/gpfs/scratch/${USER}/X/" || {
            echo "GPFS sick at chunk $chunk; pausing 60s and retrying once..."
            sleep 60
            probe_gpfs "/gpfs/scratch/${USER}/X/" || exit 1
        }
    fi
    cp "$file" /gpfs/scratch/...
    ((chunk++))
done
```

This catches mid-job stalls, gives one retry, then bails.

---

## 6. SLURM-specific patterns

### Bail-fast at job startup

The template uses this idiom:

```bash
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash $HOME/scripts/warmup_page_cache.sh "$CONDA_PREFIX"
```

`--kill-on-bad-exit=1` — if **any** node's `srun` step returns non-zero,
SLURM kills all remaining nodes' steps and the job exits. Combined with
the probe inside `warmup_page_cache.sh` (which exits 1 on probe fail),
this means: any single sick node aborts the entire multi-node job at
startup, instead of waiting 900 seconds in NCCL TCPStore rendezvous.

### What this saves you

Without probe + `--kill-on-bad-exit`:
```
[T+0s]   Job starts on 16 nodes; 1 node has stuck GPFS
[T+5s]   15 nodes import torch successfully, hit rendezvous barrier
[T+5s]   1 sick node: import torch hangs (D-state on libtorch.so stat)
[T+905s] NCCL TCPStore times out (900s default)
[T+905s] All ranks crash with "DistStoreError"
[T+906s] Job ends as FAILED, ate 16 × 905s × 4 GPU = 16 GPU-hours
```

With probe + `--kill-on-bad-exit`:
```
[T+0s]   Job starts on 16 nodes
[T+10s]  warmup_page_cache.sh probes — sick node fails
[T+11s]  --kill-on-bad-exit=1 cancels all other nodes
[T+12s]  Job ends as FAILED, ate 16 × 12s × 4 GPU = 0.21 GPU-hours
```

**~75× cheaper failure.**

---

## 7. What to do when GPFS is actually hung

If you're staring at a hung `cd` / `ls` / `mkdir` right now:

1. **Don't `kill -9`** — it won't work, and you'll waste a minute.
2. **Open a new shell** (the old one is held hostage by the syscall).
3. **Wait 5-10 minutes** — most GPFS stalls self-resolve when the
   metadata server catches up.
4. **Check broader symptoms**: `squeue -u $USER` (if SLURM is also
   responding slowly, it's a cluster-wide issue, not just you).
5. **Don't submit new jobs during a hang** — they'll queue but their
   prolog will hang too, eating queue position.
6. **If still hung after 30 min**: check `~/.bashrc` env vars (proxy IPs
   do change rarely), then file a BSC support ticket if widespread.

### What you cannot do

- **Force-unmount and remount GPFS** — root-only.
- **Terminate the D-state process** — see § 1, it's untrappable.
- **Recover unwritten data** — anything that didn't fsync before the
  hang is gone if the kernel later returns an error.

---

## 8. References

- In-house probe: `$HOME/scripts/warmup_page_cache.sh`
- SLURM template integration: see § 6 above; the template's "Page-cache
  warmup" step uses this pattern.
- BSC support context: GPFS stalls are a known platform issue; BSC
  acknowledges them but rarely root-cause-fixes individual incidents.
- Related team incident: 2026-04-17 — a teammate hit a multi-hour GPFS
  hang at 23:00 Spain time while doing `mkdir` under
  `/gpfs/scratch/ehpc1003/<user>/`. Resolved by waiting until morning.
