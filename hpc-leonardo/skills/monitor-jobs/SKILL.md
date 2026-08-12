---
name: monitor-jobs
description: Use when the user asks about SLURM job status on LEONARDO ("how is my job?", "什么时候开始", "what's the loss?", "did 40015112 finish?", "刷新一下 jobs", "为什么 fail 了"). Reports state-specific info (start ETA / live step+loss / final stats / failure root cause) and refreshes the project's `runs/RUNNING.tsv` / `HISTORY.tsv`.
---

# Monitor SLURM Jobs

## Overview

For a given job (or all the user's jobs), produce **state-specific
output** and update the project's job-tracking files. Don't dump raw
`squeue` output — distill it to what matters per state.

## Steps

1. **Identify target jobs** — scope depends on whether the current
   project has `runs/RUNNING.tsv`:

   | Situation | Scope |
   |---|---|
   | User named a specific job ID | Just that one |
   | `runs/RUNNING.tsv` **exists** | **Only** the job IDs listed in `RUNNING.tsv`. Ignore other jobs the user has running (those are tests / unrelated; user already curated what to track). |
   | `runs/RUNNING.tsv` doesn't exist | Fall back to `squeue -u $USER` (show all the user's jobs). |
   | User explicitly says "all jobs" / "all running" / "全部" | `squeue -u $USER` regardless of RUNNING.tsv |

   Reading RUNNING.tsv: skip header line; first tab-separated field
   is the JobID. Format spec: `../../references/runs-tsv-format.md`.

   **What gets WRITTEN back to RUNNING.tsv / HISTORY.tsv** is still
   `normal` jobs only — those files exist for production runs.
   `boost_qos_dbg` jobs may be reported (when seen via squeue) but are
   never added to RUNNING.tsv and never moved to HISTORY.tsv.

2. **For each job, classify by state** and follow the matching branch:

   ```
   PENDING    → see § PENDING
   RUNNING    → see § RUNNING
   COMPLETED  → see § COMPLETED
   FAILED / TIMEOUT / OUT_OF_MEMORY / CANCELLED+ / NODE_FAIL → see § FAILED
   ```

3. **Update `runs/RUNNING.tsv` / `runs/HISTORY.tsv`** — primary side-effect.
   Format spec: `../../references/runs-tsv-format.md`. **normal jobs only.**
   - For each normal job ID in `RUNNING.tsv`:
     - If still PENDING / RUNNING → **refresh** its row in-place
       (update `State` / `Step` / `Loss` / `LastChecked` cols).
     - If COMPLETED / FAILED / TIMEOUT / CANCELLED+ / NODE_FAIL →
       **MOVE the row** from `RUNNING.tsv` to `HISTORY.tsv` per the
       move-row snippet in `../../references/runs-tsv-format.md`. Fill the
       outcome columns: `Ended`, `State`, `Runtime`, `FinalLoss`,
       `Cause` (cause = signature from § FAILED, or `-` if COMPLETED).
   - For boost_qos_dbg / boost_qos_dbg jobs (even if user passes the ID):
     report state, **do not** write to RUNNING.tsv / HISTORY.tsv.
   - If `runs/` doesn't exist, initialize it with the empty RUNNING.tsv + HISTORY.tsv headers from `../../references/runs-tsv-format.md`.
   - If user passes an normal job ID that isn't in `RUNNING.tsv`,
     just report state — don't auto-add. (User decides if it's worth tracking.)

4. **Report concisely** to the user (see § Output format).

---

## § PENDING

```bash
squeue -j <id> -o "%i %j %T %r %N" -h        # state, reason, nodes
squeue -j <id> --start -o "%S" -h            # predicted start (or "N/A")
sprio -j <id> 2>/dev/null                    # priority breakdown
```

Report:
- Reason (`Priority` / `Resources` / `AssocGrpJobsLimit` / `Dependency` …)
- ETA: `--start` value (absolute time + relative offset).
- If `--start` is `N/A`: SLURM hasn't predicted yet — say so, don't lie.
- If reason hints at a problem (`AssocGrpJobsLimit` for boost_qos_dbg = the
  user already has 1 boost_qos_dbg running; explain).

---

## § RUNNING

Compute the file paths from the job's name and ID:

```bash
LOG_OUT=$SCRATCH/slurm_logs/${JOB_NAME}_${JOB_ID}.out
LOG_ERR=$SCRATCH/slurm_logs/${JOB_NAME}_${JOB_ID}.err
```

(Path source: `../../references/storage-layout.md` § 6.)

### Depth by QoS

| QoS | Tail depth | Step/Loss extraction | GPU util |
|---|---|---|---|
| `boost_qos_dbg` | `tail -1000` | last **5** (step, loss) pairs → show as trend | 3 samples, 10s apart |
| `normal` | `tail -200` | last **1** (step, loss) pair | single snapshot |

The idea: debug runs are short (≤30 min, hard cap) and you're watching them
closely → detail is cheap and useful. Production runs are long (up to 24 h on
`normal`, up to 4 days on `boost_qos_lprod`) and stable → a quick snapshot is
enough; checking too often wastes your time and the user's.

Extract:
- **Elapsed runtime**: `squeue -j <id> -o "%M" -h`
- **Walltime remaining**: `squeue -j <id> -o "%L" -h`
- **Latest step / loss**: tail per the depth above, grep for the project's
  log format. Common patterns to try, in order:
  ```bash
  grep -E '(step|Step|iter|Iter)[ :=]+[0-9]+'  $LOG_OUT | tail -3
  grep -iE 'loss[ :=]+[-0-9.eE]+'              $LOG_OUT | tail -3
  grep -E '\bit/s\b|\bs/it\b'                   $LOG_OUT | tail -3   # tqdm rate
  ```
  Don't fabricate values. If logs don't have anything matchable, say
  "no recognizable step/loss in last 200 lines" — that's useful info,
  not a failure to extract.
- **Step rate**: from two consecutive step timestamps if available, or
  tqdm `it/s`. Round to 0.1 s/step or 0.01 it/s. If can't compute, omit.
- **ETA to completion** (best-effort): if you have step rate AND total
  steps (e.g. `[100/5000]`), project remaining wall-clock and compare
  to walltime remaining.
- **`.err` tail**: `tail -20 $LOG_ERR`. Note any WARNING/ERROR lines but
  don't panic — NCCL/torch print warnings constantly. Only flag actual
  errors (`Traceback`, `RuntimeError`, `CUDA error`, `OOM`).

If `.out` / `.err` doesn't exist yet (job started < 30 sec ago),
report "logs not yet flushed".

### GPU utilization — depth depends on QoS

Different sampling strategy by QoS — `boost_qos_dbg` = deep dive (short jobs, frequent samples), `normal` / `boost_qos_lprod` / `boost_qos_bprod` = quick glance (long stable jobs).

#### `boost_qos_dbg` (debugging → multi-sample to see trend)

```bash
# 3 samples, 10s apart, to expose ramp-up / stalls / oscillation
for i in 1 2 3; do
    srun --jobid=<id> --overlap --ntasks=1 --nodes=1 \
         nvidia-smi --query-gpu=utilization.gpu \
                    --format=csv,noheader,nounits 2>/dev/null \
        | tr '\n' ',' | sed 's/,$/\n/'
    [ $i -lt 3 ] && sleep 10
done
```

Report 3 lines:
```
GPU util:
  t+0s   88,90,87,89
  t+10s  91,92,86,88
  t+20s  90,89,88,89
```

#### `normal` (production → single snapshot)

```bash
srun --jobid=<id> --overlap --ntasks=1 --nodes=1 \
     nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null
```

Report as one line: `GPU util:  88,90,87,89`

#### Both modes

**Don't classify, don't diagnose, don't poll beyond the sampling above.**
User reads the numbers and decides. Re-running monitor-jobs gives a
fresh sample.

---

## § COMPLETED

```bash
sacct -j <id> -o JobID,State,ExitCode,Elapsed,NodeList,Start,End -P --noheader -X
```

Report:
- Exit code (must be `0:0` for success).
- Total runtime.
- Last loss (grep `.out` tail same way as RUNNING).
- Average step rate (if extractable).
- Final ckpt path (look for `Saved to /...` or `output_dir` mentions in
  the last 200 lines of `.out`).

---

## § FAILED (also TIMEOUT / OOM / CANCELLED+ / NODE_FAIL)

This is the high-value branch — root-cause attribution.

1. **Get exit context**:
   ```bash
   sacct -j <id> -o JobID,State,ExitCode,DerivedExitCode,Elapsed,NodeList,Start,End,Reason -P --noheader -X
   ```

2. **Read the .err first** (most failures land there):
   ```bash
   tail -100 $LOG_ERR
   ```

3. **Then .out tail**:
   ```bash
   tail -100 $LOG_OUT
   ```

4. **Match against the failure-signature table** below — return the
   FIRST match that fits, with the verbatim line numbers from the log
   so the user can grep.

### Failure signatures (priority order)

| Signature in log | Likely root cause | Hint |
|---|---|---|
| `out of memory` / `CUDA out of memory` | OOM in CUDA | Reduce batch size / enable gradient checkpointing / ZeRO-3 |
| `Killed` (no traceback) + exit `137` | OS / cgroup OOM-killed (CPU memory) | Check `--mem` if set, reduce dataloader workers |
| `Killed` + exit `143` | SIGTERM from SLURM (walltime hit) | Either job hit `--time` or scancel |
| `signal 9` / `SIGKILL` | Forcibly killed (walltime, OOM, node fail) | Check sacct `Reason` field |
| `NCCL ... timeout` / `DistStoreError` / `TCPStore` | Multi-node rendezvous failed | See `../../references/lustre-resilience.md` — likely cold-import Lustre storm; verify warmup ran |
| `Traceback (most recent call last)` | Python exception | Read the actual exception message at the bottom of the traceback |
| `ModuleNotFoundError` / `ImportError` | Wrong conda env | Check `conda activate <env>` matches what's installed |
| `permission denied` / `Permission denied` | Path / quota | Check the offending path; check `quota -s` |
| `No such file or directory` | Missing path / Lustre not mounted on this node | Re-check absolute paths; rare Lustre hiccup |
| `Bus error` / `Segmentation fault` | Native crash (often CUDA driver / NCCL) | Re-run; if persistent, email `superc@cineca.it` |
| `slurmstepd: error: *** PROLOG FAILED` | Node prolog failed (often Lustre hang) | See `../../references/lustre-resilience.md` |
| sacct State `NODE_FAIL` | Compute node died mid-job | Re-submit; nothing the user did wrong |

If nothing matches: report "couldn't auto-classify; first ERROR line
in `.err`: <line>" and let the user investigate.

---

## Output format

### Single job → per-job block (full detail)

```
Job <id>  <name>  <STATE> on <node-or-->
  Reason:    <only-if-PENDING>            ← omit otherwise
  Started:   <abs-time> (<rel> ago)       ← RUNNING+
  Runtime:   <Dd Hh Mm>                   ← RUNNING+
  Walltime:  <left> / <total>             ← RUNNING+
  Step:      <N> / <total?>               ← if extractable
  Loss:      <latest>                     ← if extractable
  Rate:      <s/step or it/s>             ← if extractable
  GPU util:  <a,b,c,d>                    ← RUNNING only
  ETA:       <projected-finish>           ← if rate+total available
  Result:    ✅/❌ exit <code>             ← COMPLETED/FAILED
  Cause:     <signature-match summary>    ← FAILED only
  Logs:      <out path> / <err path>      ← FAILED only (so user can grep)
```

### Multiple jobs → summary table FIRST, then per-job blocks

When reporting on **2 or more** jobs (the common case when user runs
monitor-jobs against `runs/RUNNING.tsv`), lead with a compact summary
**markdown table** so the user sees everything at a glance — Claude
Code's chat renderer turns it into a visual framed table.

```
| JobID    | Name        | State     | Node      | Runtime | Step | Loss   | GPU util       | Detail                |
|----------|-------------|-----------|-----------|---------|------|--------|----------------|-----------------------|
| 40015112 | 1n3d        | RUNNING   | as04r5b23 | 3h 12m  | 234  | 0.412  | 88,90,87,89    | walltime 22h left     |
| 40009999 | wan22vae_8n | PENDING   | -         | -       | -    | -      | -              | starts ~2h (Priority) |
| 40059650 | slack_test  | COMPLETED | -         | 1m 0s   | -    | -      | -              | exit 0                |
| 40012345 | s2v_lora_4n | FAILED    | -         | 1h 23m  | -    | -      | -              | OOM @ step 234        |
```

Rules for the summary table:
- One row per job; columns: `JobID Name State Node Runtime Step Loss "GPU util" Detail`.
- Use `-` for fields that don't apply to that state.
- `GPU util` shows the 4-GPU comma-separated snapshot (RUNNING only;
  for `boost_qos_dbg` jobs, just put the LATEST sample here — the trend
  goes in the per-job block below).
- `Detail` summarizes the state-specific extra:
  - PENDING: predicted start ETA + reason
  - RUNNING: walltime remaining (or `ETA <abs-time>` if rate available)
  - COMPLETED: `exit <code>`
  - FAILED: failure signature first match (e.g. `OOM @ step 234`)
- Keep cells short — don't include log paths in the table; put those
  in the per-job blocks below.

After the table, show per-job blocks for:
- **All FAILED jobs** — give the user log paths to grep.
- **All RUNNING `boost_qos_dbg` jobs** — give the GPU-util time series and
  step/loss trend (5 samples) for debug visibility.
- RUNNING `normal` / COMPLETED / PENDING jobs: table is enough,
  no extra block.

---

## Common Mistakes — DO NOT MAKE THESE

| Mistake | Right way |
|---|---|
| Reporting "step 100, loss 0.234" without verifying it's the LATEST line in the log | Always `tail -200` then grep, take the LAST match |
| Saying "job will finish in 2h" without confirming step rate is stable | Only report ETA if recent rate is consistent (tail-3 matches) |
| Auto-cancelling a stuck job | NEVER. Report the symptom; user decides scancel |
| Re-submitting failed jobs without telling the user | NEVER. Report failure cause; user decides re-submit |
| Creating `runs/` outside a project (e.g. `~/runs/`) | Only update `runs/` if the user is in a project that already has it |
| Polling on a loop (`while true; squeue`) | NEVER. Single snapshot; user runs again to refresh |
| Reading 10000-line `.out` files in full | `tail -200` is enough for live; `tail -500` for FAILED root-cause |

## Red Flags — STOP and reconsider

- "User said the job is bad — let me scancel it" (NO. Report, don't act.)
- "I'll re-submit since it failed obviously" (NO. User decides.)
- "Loss is 0.234 — wait that's from the start; let me make up a recent one" (NO. Say what's actually in the log.)
- "Logs hung when I tried to read — let me ignore" (NO. Lustre hang is a signal — see `lustre-resilience.md`, report and bail.)
- "Walltime is short — I'll suggest extending via scontrol update" (NO. normal/debug walltimes are user-set; suggest re-submit instead.)

## What this skill does NOT do

- Cancel jobs (`scancel`).
- Re-submit failed jobs.
- Modify SLURM job parameters via `scontrol update`.
- Tail logs continuously.
- Compute live histograms / plots.
- Decide whether a failure is "transient" or "permanent" — report the
  signature; user decides.

## Background References

- `../../references/slurm.md` — QoS limits, sacct/squeue command reference,
  failure-mode table (§ 12)
- `../../references/storage-layout.md` § 6 — log path location (`slurm_logs/`)
- `../../references/lustre-resilience.md` — when log reads themselves hang
- `../../references/runs-tsv-format.md` — RUNNING.tsv / HISTORY.tsv column spec
  + parse / append / move snippets
