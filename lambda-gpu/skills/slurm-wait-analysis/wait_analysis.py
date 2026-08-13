#!/usr/bin/env python3
"""Analyze SLURM job wait time and run time, emit a Markdown and/or HTML report.

Wait = Start - Submit (queue time). Run time = SLURM's Elapsed.
Filters: by node count and/or minimum run time. See SKILL.md for usage.
"""
import argparse
import datetime
import os
import subprocess
import sys


def parse_time(t):
    if t in ("None", "Unknown", ""):
        return None
    return datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S")


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def elapsed_sec(e):
    """Parse SLURM Elapsed (HH:MM:SS or D-HH:MM:SS) into seconds."""
    d = 0
    if "-" in e:
        dd, e = e.split("-")
        d = int(dd)
    h, m, s = map(int, e.split(":"))
    return d * 86400 + h * 3600 + m * 60 + s


def fetch_jobs(user, start, end):
    cmd = [
        "sacct", "-u", user, "--starttime", start,
        "--format=JobID,NNodes,Submit,Start,End,State,Elapsed", "-X", "-P",
    ]
    if end:
        cmd += ["--endtime", end]
    out = subprocess.check_output(cmd).decode()
    rows = []
    for line in out.strip().splitlines()[1:]:
        jid, nn, sub, start_t, end_t, state, elapsed = line.split("|")
        rows.append({
            "jid": jid, "nnodes": int(nn) if nn.isdigit() else 0,
            "submit": sub, "start": start_t, "state": state, "elapsed": elapsed,
            "wait_sec": None, "run_sec": elapsed_sec(elapsed),
        })
        st = parse_time(start_t)
        s = parse_time(sub)
        if st is not None and s is not None:
            rows[-1]["wait_sec"] = int((st - s).total_seconds())
    return rows


def apply_filters(rows, nodes, min_runtime_h):
    out = []
    for r in rows:
        if nodes is not None and r["nnodes"] != nodes:
            continue
        if min_runtime_h is not None and r["run_sec"] < min_runtime_h * 3600:
            continue
        out.append(r)
    return out


def state_class(s):
    for key in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "RUNNING", "PENDING"):
        if key in s:
            return key.lower()
    return ""


def stats(rows):
    waits = [r["wait_sec"] for r in rows if r["wait_sec"] is not None]
    runs = [r["run_sec"] for r in rows if r["run_sec"] > 0]
    return {
        "n": len(rows),
        "avg_wait": sum(waits) / len(waits) if waits else 0,
        "avg_run": sum(runs) / len(runs) if runs else 0,
        "max_wait": max(waits) if waits else 0,
        "n_waits": len(waits),
    }


def filter_label(nodes, min_runtime_h):
    parts = []
    if nodes is not None:
        parts.append(f"NNodes = {nodes}")
    if min_runtime_h is not None:
        parts.append(f"run time ≥ {min_runtime_h:g} h")
    return " and ".join(parts) if parts else "none"


def render_md(rows, st, nodes, min_runtime_h, user, start, gen):
    flt = filter_label(nodes, min_runtime_h)
    L = []
    title = "SLURM Job Wait & Run Time Analysis"
    if flt != "none":
        title += f" ({flt})"
    L.append(f"# {title}\n")
    L.append(f"**User:** {user} &nbsp; | &nbsp; **Window:** since {start} &nbsp; | "
             f"&nbsp; **Generated:** {gen} &nbsp; | &nbsp; **Filter:** {flt} &nbsp; | "
             f"&nbsp; **Jobs:** {st['n']}\n")
    L.append("## Summary\n")
    L.append("| Metric | Value |")
    L.append("|--------|-------|")
    L.append(f"| Total jobs | {st['n']} |")
    L.append(f"| Avg wait (started jobs) | {hms(st['avg_wait'])} ({st['avg_wait']/3600:.2f} h) |")
    L.append(f"| Avg run time | {hms(st['avg_run'])} ({st['avg_run']/3600:.2f} h) |")
    L.append(f"| Max wait | {hms(st['max_wait'])} ({st['max_wait']/3600:.1f} h) |")
    L.append("")
    L.append("## Jobs\n")
    L.append("| Job ID | Nodes | Submit Time | Start Time | State | Wait (sec) | Wait (hh:mm:ss) | Run Time (hh:mm:ss) |")
    L.append("|--------|------:|-------------|------------|-------|-----------:|----------------:|--------------------:|")
    for r in rows:
        wstr = hms(r["wait_sec"]) if r["wait_sec"] is not None else "—"
        wnum = f"{r['wait_sec']:,}" if r["wait_sec"] is not None else "—"
        startd = r["start"] if r["start"] not in ("None", "Unknown") else "—"
        rstr = hms(r["run_sec"]) if r["elapsed"] != "00:00:00" else "—"
        L.append(f"| {r['jid']} | {r['nnodes']} | {r['submit']} | {startd} | {r['state']} | {wnum} | {wstr} | {rstr} |")
    L.append(f"| **AVG** | | | | | **{st['avg_wait']:,.0f}** | "
             f"**{hms(st['avg_wait'])}** ({st['avg_wait']/3600:.2f} h) | "
             f"**{hms(st['avg_run'])}** ({st['avg_run']/3600:.2f} h) |")
    L.append("")
    L.append("> **Notes:** Wait = Start − Submit. Jobs that never started (Start = —) are excluded from the wait average.")
    L.append("> Run Time is SLURM's `Elapsed`; entries near `24:10:xx` are 1 day + 10 min, i.e. jobs that hit the 24 h wall-clock limit (shown as `TIMEOUT`).")
    return "\n".join(L) + "\n"


def render_html(rows, st, nodes, min_runtime_h, user, start, gen):
    flt = filter_label(nodes, min_runtime_h)
    title = "SLURM Job Wait &amp; Run Time Analysis"
    trs = []
    for r in rows:
        wstr = hms(r["wait_sec"]) if r["wait_sec"] is not None else "&mdash;"
        wnum = f"{r['wait_sec']:,}" if r["wait_sec"] is not None else "&mdash;"
        startd = r["start"] if r["start"] not in ("None", "Unknown") else "&mdash;"
        rstr = hms(r["run_sec"]) if r["elapsed"] != "00:00:00" else "&mdash;"
        cls = state_class(r["state"])
        trs.append(f"""    <tr class="{cls}">
      <td class="mono">{r['jid']}</td><td class="num">{r['nnodes']}</td>
      <td class="mono">{r['submit']}</td><td class="mono">{startd}</td>
      <td><span class="badge {cls}">{r['state']}</span></td>
      <td class="num">{wnum}</td><td class="num mono">{wstr}</td><td class="num mono">{rstr}</td>
    </tr>""")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; margin:0; padding:2rem; background:#f5f6f8; color:#1f2933; }}
  h1 {{ font-size:1.5rem; margin:0 0 .25rem; }}
  .sub {{ color:#62707f; margin:0 0 1.5rem; font-size:.9rem; }}
  .cards {{ display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem; }}
  .card {{ background:#fff; border-radius:10px; padding:1rem 1.25rem; box-shadow:0 1px 3px rgba(0,0,0,.08); flex:1; min-width:160px; }}
  .card .label {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:#7b8794; }}
  .card .value {{ font-size:1.4rem; font-weight:600; margin-top:.25rem; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:10px; overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,.08); font-size:.875rem; }}
  th,td {{ padding:.55rem .75rem; text-align:left; border-bottom:1px solid #eef1f4; }}
  th {{ background:#2d3748; color:#fff; font-weight:600; font-size:.8rem; }}
  tbody tr:hover {{ background:#f0f4ff; }}
  .mono {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:.82rem; }}
  .num {{ text-align:right; }}
  .badge {{ display:inline-block; padding:.15rem .5rem; border-radius:12px; font-size:.72rem; font-weight:600; }}
  .badge.completed {{ background:#d4f4dd; color:#1a7f37; }}
  .badge.failed {{ background:#fcd9d9; color:#b42318; }}
  .badge.timeout {{ background:#fde9c8; color:#b54708; }}
  .badge.cancelled {{ background:#e4e7eb; color:#52606d; }}
  .badge.running {{ background:#d6e4ff; color:#1d4ed8; }}
  .badge.pending {{ background:#ede9fe; color:#6d28d9; }}
  tfoot td {{ font-weight:700; background:#f7f9fb; border-top:2px solid #2d3748; }}
  .legend {{ margin-top:1rem; font-size:.78rem; color:#62707f; }}
</style></head><body>
  <h1>{title}</h1>
  <p class="sub">User: {user} &nbsp;|&nbsp; Window: since {start} &nbsp;|&nbsp; Generated {gen} &nbsp;|&nbsp; Filter: {flt} &nbsp;|&nbsp; {st['n']} jobs</p>
  <div class="cards">
    <div class="card"><div class="label">Total Jobs</div><div class="value">{st['n']}</div></div>
    <div class="card"><div class="label">Avg Wait (started)</div><div class="value">{st['avg_wait']/3600:.2f} h</div></div>
    <div class="card"><div class="label">Avg Run Time</div><div class="value">{st['avg_run']/3600:.2f} h</div></div>
    <div class="card"><div class="label">Max Wait</div><div class="value">{st['max_wait']/3600:.1f} h</div></div>
  </div>
  <table><thead><tr>
    <th>Job ID</th><th class="num">Nodes</th><th>Submit Time</th><th>Start Time</th><th>State</th>
    <th class="num">Wait (sec)</th><th class="num">Wait (hh:mm:ss)</th><th class="num">Run Time (hh:mm:ss)</th>
  </tr></thead><tbody>
{chr(10).join(trs)}
  </tbody><tfoot><tr>
    <td colspan="5">AVG (over started / run jobs)</td>
    <td class="num">{st['avg_wait']:,.0f}</td>
    <td class="num mono">{hms(st['avg_wait'])} ({st['avg_wait']/3600:.2f} h)</td>
    <td class="num mono">{hms(st['avg_run'])} ({st['avg_run']/3600:.2f} h)</td>
  </tr></tfoot></table>
  <p class="legend">Wait = Start &minus; Submit. Jobs that never started (Start = &mdash;) are excluded from the wait average.
  Run Time is SLURM's Elapsed; entries near <span class="mono">24:10:xx</span> are 1&nbsp;day&nbsp;+&nbsp;10&nbsp;min, i.e. hit the 24h wall-clock limit.</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser(description="SLURM wait/run time report")
    ap.add_argument("--user", default=os.environ.get("USER", "user"))
    ap.add_argument("--start", default="2026-05-26", help="sacct --starttime (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="sacct --endtime (optional)")
    ap.add_argument("--nodes", type=int, default=None, help="keep only jobs with this NNodes")
    ap.add_argument("--min-runtime", type=float, default=None,
                    help="keep only jobs with run time >= this many hours")
    ap.add_argument("--format", choices=["md", "html", "both"], default="md")
    ap.add_argument("--output", default="slurm_wait_analysis",
                    help="output path WITHOUT extension (extension added per format)")
    ap.add_argument("--generated", default=None, help="generation date label (default: today)")
    args = ap.parse_args()

    rows = fetch_jobs(args.user, args.start, args.end)
    rows = apply_filters(rows, args.nodes, args.min_runtime)
    if not rows:
        sys.exit("No jobs matched the filters.")
    st = stats(rows)
    gen = args.generated or datetime.date.today().isoformat()

    written = []
    if args.format in ("md", "both"):
        p = args.output + ".md"
        open(p, "w").write(render_md(rows, st, args.nodes, args.min_runtime, args.user, args.start, gen))
        written.append(p)
    if args.format in ("html", "both"):
        p = args.output + ".html"
        open(p, "w").write(render_html(rows, st, args.nodes, args.min_runtime, args.user, args.start, gen))
        written.append(p)

    print("wrote: " + ", ".join(written))
    print(f"{st['n']} jobs | avg wait {st['avg_wait']/3600:.2f} h | "
          f"avg run {st['avg_run']/3600:.2f} h | max wait {st['max_wait']/3600:.1f} h")


if __name__ == "__main__":
    main()
