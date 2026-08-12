# `runs/RUNNING.tsv` & `runs/HISTORY.tsv` — Canonical Format

The shared format both `create-slurm-job-script` and `monitor-jobs`
read/write. **Don't drift.** If you change column order or names,
update this doc and both skills.

**Related**: `storage-layout.md` (per-project layout); CLAUDE.md "Job
Tracking" rule (when to write).

## Why TSV (not Markdown / CSV)

- **TSV** = tab-separated values. One row per job.
- Easy to append: `printf '%s\t%s\t...\n' >> file`
- Easy to parse: `awk -F'\t'`
- Easy to view aligned: `column -t -s$'\t' runs/RUNNING.tsv`
- No quoting headaches with commas (which CSV would need for the
  `Cause` field).

## Scope

Track **`normal` (production) jobs only**. `boost_qos_dbg` and
`boost_qos_dbg` are short-lived tests — don't pollute RUNNING.tsv
with them.

## File location

```
<project>/runs/RUNNING.tsv
<project>/runs/HISTORY.tsv
```

`runs/` is project-local. Add `runs/` to the project's `.gitignore`
(already in `~/scripts/lib/templates/gitignore-ai-tools`).

## RUNNING.tsv columns (10, tab-separated)

Header line (literal — note tabs between fields, not pipes):
```
JobID<TAB>Script<TAB>QoS<TAB>Nodes<TAB>Time<TAB>Submitted<TAB>State<TAB>Step<TAB>Loss<TAB>LastChecked
```

When viewed with `column -t -s$'\t'`:
```
JobID     Script             QoS       Nodes  Time        Submitted         State    Step  Loss    LastChecked
40015112  train/quick_1n.sh  normal  1      3-00:00:00  2026-05-05 11:33  RUNNING  234   0.4123  2026-05-06 16:30
```

| # | Column | Meaning |
|---|---|---|
| 1 | `JobID` | SLURM job ID (from `sbatch` output) |
| 2 | `Script` | Path relative to project root (`train/quick_1n.sh`) |
| 3 | `QoS` | `normal` |
| 4 | `Nodes` | node count |
| 5 | `Time` | walltime (`HH:MM:SS` or `D-HH:MM:SS`) |
| 6 | `Submitted` | `YYYY-MM-DD HH:MM` (local time) |
| 7 | `State` | `PENDING` / `RUNNING` |
| 8 | `Step` | latest training step (or `-`) |
| 9 | `Loss` | latest loss (or `-`) |
| 10 | `LastChecked` | `YYYY-MM-DD HH:MM` of last `monitor-jobs` refresh |

## HISTORY.tsv columns (11, tab-separated)

Replaces RUNNING's `Step / Loss / LastChecked` with `Ended / Runtime /
FinalLoss / Cause`:

```
JobID  Script  QoS  Nodes  Time  Submitted  Ended  State  Runtime  FinalLoss  Cause
```

Pretty-printed example:
```
JobID     Script             QoS       Nodes  Time        Submitted         Ended             State      Runtime  FinalLoss  Cause
40015112  train/quick_1n.sh  normal  1      3-00:00:00  2026-05-05 11:33  2026-05-08 11:33  COMPLETED  3-00:00  0.0421     -
40012345  train/test_4n.sh   normal  4      1-00:00:00  2026-05-04 22:00  2026-05-04 23:30  FAILED     1h 30m   1.234      OOM at step 234 (.err line 67)
```

## Pretty-print for humans

```bash
column -t -s$'\t' runs/RUNNING.tsv
column -t -s$'\t' runs/HISTORY.tsv
```

## Initialize empty RUNNING.tsv

```bash
mkdir -p runs
[ -f runs/RUNNING.tsv ] || printf '%s\n' \
    $'JobID\tScript\tQoS\tNodes\tTime\tSubmitted\tState\tStep\tLoss\tLastChecked' \
    > runs/RUNNING.tsv
```

## Initialize empty HISTORY.tsv

```bash
[ -f runs/HISTORY.tsv ] || printf '%s\n' \
    $'JobID\tScript\tQoS\tNodes\tTime\tSubmitted\tEnded\tState\tRuntime\tFinalLoss\tCause' \
    > runs/HISTORY.tsv
```

## Append a new RUNNING row (after `sbatch`)

Drop-in shell snippet for `normal` jobs only:

```bash
# $1 = path to the .sh script you're about to submit
SCRIPT="$1"
JOB_ID=$(sbatch "$SCRIPT" | awk '{print $4}')
[ -z "$JOB_ID" ] && { echo "sbatch failed"; exit 1; }

QOS=$(grep -oP '^#SBATCH --qos=\K\S+'   "$SCRIPT")
N=$(  grep -oP '^#SBATCH --nodes=\K\d+' "$SCRIPT")
T=$(  grep -oP '^#SBATCH --time=\K\S+'  "$SCRIPT")
NOW=$(date '+%Y-%m-%d %H:%M')
NAME=$(basename "$SCRIPT")

mkdir -p runs
[ -f runs/RUNNING.tsv ] || printf '%s\n' \
    $'JobID\tScript\tQoS\tNodes\tTime\tSubmitted\tState\tStep\tLoss\tLastChecked' \
    > runs/RUNNING.tsv

printf '%s\t%s\t%s\t%s\t%s\t%s\tPENDING\t-\t-\t%s\n' \
    "$JOB_ID" "$NAME" "$QOS" "$N" "$T" "$NOW" "$NOW" \
    >> runs/RUNNING.tsv

echo "Submitted $JOB_ID and tracked in runs/RUNNING.tsv"
```

## Refresh a row in-place (RUNNING.tsv)

```bash
# $1=JOB_ID, $2=state, $3=step, $4=loss
awk -F'\t' -v OFS='\t' \
    -v j="$1" -v st="$2" -v sp="$3" -v ls="$4" \
    -v now="$(date '+%Y-%m-%d %H:%M')" \
    '$1 == j {$7 = st; $8 = sp; $9 = ls; $10 = now} {print}' \
    runs/RUNNING.tsv > runs/RUNNING.tsv.tmp \
    && mv runs/RUNNING.tsv.tmp runs/RUNNING.tsv
```

## Move a row from RUNNING.tsv to HISTORY.tsv (job ended)

```bash
# $1=JOB_ID, $2=state (COMPLETED/FAILED/...), $3=runtime, $4=final_loss (or -), $5=cause (or -)
JOB_ID="$1"; STATE="$2"; RUNTIME="$3"; LOSS="$4"; CAUSE="$5"
NOW_END=$(date '+%Y-%m-%d %H:%M')

[ -f runs/HISTORY.tsv ] || printf '%s\n' \
    $'JobID\tScript\tQoS\tNodes\tTime\tSubmitted\tEnded\tState\tRuntime\tFinalLoss\tCause' \
    > runs/HISTORY.tsv

# 1. Reformat the row from RUNNING and append to HISTORY
awk -F'\t' -v OFS='\t' -v j="$JOB_ID" -v end="$NOW_END" \
    -v st="$STATE" -v rt="$RUNTIME" -v fl="$LOSS" -v cz="$CAUSE" \
    '$1 == j {print $1, $2, $3, $4, $5, $6, end, st, rt, fl, cz}' \
    runs/RUNNING.tsv >> runs/HISTORY.tsv

# 2. Remove the row from RUNNING
awk -F'\t' -v j="$JOB_ID" '$1 != j {print}' \
    runs/RUNNING.tsv > runs/RUNNING.tsv.tmp \
    && mv runs/RUNNING.tsv.tmp runs/RUNNING.tsv
```

## Parse rule (skills reading RUNNING.tsv)

- Skip line 1 (header — first field is literal `JobID`).
- For each remaining line, the first tab-separated field is the JobID.
  ```bash
  tail -n +2 runs/RUNNING.tsv | while IFS=$'\t' read -r jid script qos n t submit state step loss last; do
      echo "Job $jid → $state, step=$step loss=$loss"
  done
  ```

## Don't

- ❌ Track `boost_qos_dbg` jobs (too noisy; they're 2-hour tests).
- ❌ Hand-edit RUNNING.tsv / HISTORY.tsv mid-experiment — let the skills.
- ❌ Reorder columns; both skills depend on this exact order.
- ❌ Use literal tab characters inside a field value (e.g. in `Cause`)
  — that breaks parsing. Replace `\t` with space when writing.
