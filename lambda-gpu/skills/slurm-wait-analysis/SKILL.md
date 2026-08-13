---
name: slurm-wait-analysis
description: Analyze SLURM queue wait time and run time for the user's jobs and write a readable Markdown (or HTML) report with a per-job table, summary cards, and averages. Supports filtering by node count and minimum run time. Use when the user says "analyze my slurm wait times", "make a table of my job wait/run times", "update the slurm wait analysis", "only 16-node jobs", or "drop jobs shorter than 1h".
---

# Analyze SLURM job wait & run time

Query `sacct` for the user's jobs, compute **wait time** (Start − Submit, i.e. queue
time) and **run time** (SLURM `Elapsed`), and write a report table. Output defaults to
`slurm_wait_analysis.md` in the repo root (rendered nicely by VS Code's Markdown
preview, Ctrl/Cmd+Shift+V).

## How to run it

```bash
python3 .claude/skills/slurm-wait-analysis/wait_analysis.py \
  [--user <NETID>]          # default: $USER \
  [--start YYYY-MM-DD]      # sacct --starttime, default 2026-05-26 \
  [--end YYYY-MM-DD]        # optional sacct --endtime \
  [--nodes N]               # keep only jobs that allocated N nodes \
  [--min-runtime HOURS]     # keep only jobs with run time >= HOURS (float ok) \
  [--format md|html|both]   # default md \
  [--output PATH_NO_EXT]    # default slurm_wait_analysis (extension added per format) \
  [--generated YYYY-MM-DD]  # date label in the report, default today
```

It prints the output path(s) and a one-line summary (job count, avg wait, avg run,
max wait) to stdout.

## What it computes

| column | meaning |
|---|---|
| Wait (sec) / (hh:mm:ss) | `Start − Submit` — time spent queued. Jobs that never started show `—` and are excluded from the wait average. |
| Run Time (hh:mm:ss) | SLURM `Elapsed`. `D-HH:MM:SS` is normalized to `HH:MM:SS` (e.g. `1-00:10:27` → `24:10:27`). Values near 24:10 mean the job hit the 24 h wall-clock limit (`TIMEOUT`). |
| State | color-coded as a badge in the HTML output. |

The report includes summary stats (total jobs, avg wait over started jobs, avg run
time, max wait) and an `AVG` footer row.

## Procedure for the agent

1. Run the helper with the filters the user asked for. Common requests map directly to
   flags: "only 16-node jobs" → `--nodes 16`; "drop runs under an hour" →
   `--min-runtime 1`; "as HTML" → `--format html` (or `both`).
2. Report the printed summary (job count + averages) and link the output file so the
   user can open the preview.
3. **"Update the table"** = re-run with the same filters that produced the current
   `slurm_wait_analysis.md` (most recently it was `--nodes 16 --min-runtime 1`). New
   jobs since the last run will be picked up automatically.

## Notes / gotchas

- Uses `sacct -X` (one row per job, no job steps). `NNodes` is the requested/allocated
  node count; jobs cancelled before starting report `AllocNodes=0` but still carry their
  requested `NNodes`, so they pass a `--nodes` filter yet show `—` for wait.
- `--start` bounds the sacct window; widen it (e.g. `--start 2026-05-01`) for more history.
- HTML output is self-contained (inline CSS), but Leonardo login nodes are headless
  (no `DISPLAY`), so a browser won't open there — prefer Markdown for in-VS-Code viewing,
  or scp the HTML locally.
