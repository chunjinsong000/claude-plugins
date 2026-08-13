#!/usr/bin/env python3
"""Analyze one or more streaming-benchmark run dirs (each = a --save_path from
run_bench.py) and print a per-run summary + a markdown comparison table.

For each run dir it reads:
  * module_fps.csv      per-clip per-module fps  -> steady-state DiT / VAE-dec / pose fps + latency
  * gpu_telemetry.csv   nvidia-smi samples       -> mean/peak core-clock / power / temp (filtered to --gpu-index)
  * memory_report.txt   (if --mem_profile was on) -> peak device memory ("WHOLE COST")
and, unless --no-plot, renders <run>/gpu_curves.png via the repo's
live_dealer/infer/utils/plot_gpu_curves.py.

Steady-state = drop the first --warmup-clips rows of module_fps.csv (the compile
/ cold-cache clips) and take the median of the rest.

Usage:
    python3 .claude/skills/benchmark-streaming/summarize_bench.py <run_dir> [<run_dir> ...] \
        [--gpu-index 1] [--warmup-clips 2] [--no-plot] [--markdown out.md]
"""
import argparse
import csv
import os
import statistics as st
import subprocess
import sys

PLOTTER = "live_dealer/infer/utils/plot_gpu_curves.py"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [{k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
                for r in csv.DictReader(f)]


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def summarize_run(run_dir, gpu_index, warmup):
    s = {"dir": run_dir, "name": os.path.basename(run_dir.rstrip("/"))}

    # --- per-module fps (single-GPU streaming path) ---
    fps = read_csv(os.path.join(run_dir, "module_fps.csv"))
    steady = fps[warmup:] if len(fps) > warmup else fps
    s["n_clips"] = len(fps)
    s["n_steady"] = len(steady)
    for col in ("dit_fps", "vae_dec_fps", "pose_fps", "dit_ms", "vae_dec_ms", "pose_ms"):
        s[col] = med([fnum(r, col) for r in steady])

    # --- GPU telemetry ---
    tel = [r for r in read_csv(os.path.join(run_dir, "gpu_telemetry.csv"))
           if r.get("index") == str(gpu_index)]
    if tel:
        t0 = min(fnum(r, "timestamp") for r in tel)
        t1 = max(fnum(r, "timestamp") for r in tel)
        s["span_s"] = t1 - t0
        for src, key in (("power_draw_w", "power"), ("temperature_gpu_c", "temp"),
                         ("clocks_graphics_mhz", "clock")):
            vals = [fnum(r, src) for r in tel]
            vals = [v for v in vals if v is not None]
            if vals:
                s[f"{key}_mean"] = sum(vals) / len(vals)
                s[f"{key}_peak"] = max(vals)
    s["n_tel"] = len(tel)

    # --- memory report (optional) ---
    mp = os.path.join(run_dir, "memory_report.txt")
    if os.path.exists(mp):
        with open(mp) as f:
            for ln in f:
                if "WHOLE COST" in ln:
                    try:
                        s["peak_mem_gb"] = float(ln.split(":")[-1].strip().split()[0])
                    except (ValueError, IndexError):
                        pass
    return s


def fmt(v, spec=".1f"):
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def print_run(s):
    print(f"\n=== {s['name']} ===")
    print(f"  dir: {s['dir']}")
    print(f"  clips: {s['n_clips']} ({s['n_steady']} steady) | telemetry samples: {s['n_tel']}"
          + (f" over {fmt(s.get('span_s'), '.0f')}s" if s.get("span_s") else ""))
    print(f"  DiT      : {fmt(s.get('dit_fps'))} fps  ({fmt(s.get('dit_ms'))} ms)")
    print(f"  VAE dec  : {fmt(s.get('vae_dec_fps'))} fps  ({fmt(s.get('vae_dec_ms'))} ms)")
    print(f"  pose enc : {fmt(s.get('pose_fps'))} fps  ({fmt(s.get('pose_ms'))} ms)")
    if s.get("power_mean") is not None:
        print(f"  power    : {fmt(s.get('power_mean'), '.0f')} / {fmt(s.get('power_peak'), '.0f')} W (mean/peak)")
        print(f"  temp     : {fmt(s.get('temp_mean'), '.0f')} / {fmt(s.get('temp_peak'), '.0f')} °C (mean/peak)")
        print(f"  clock    : {fmt(s.get('clock_mean'), '.0f')} / {fmt(s.get('clock_peak'), '.0f')} MHz (mean/peak)")
    if s.get("peak_mem_gb") is not None:
        print(f"  peak mem : {fmt(s.get('peak_mem_gb'), '.2f')} GB (device, whole cost)")


def markdown_table(rows):
    cols = [("name", "run", "{}"), ("dit_fps", "DiT fps", "{:.1f}"),
            ("vae_dec_fps", "VAEdec fps", "{:.1f}"), ("pose_fps", "pose fps", "{:.1f}"),
            ("power_mean", "power W (mean)", "{:.0f}"), ("power_peak", "power W (peak)", "{:.0f}"),
            ("temp_peak", "temp °C (peak)", "{:.0f}"), ("clock_mean", "clock MHz", "{:.0f}"),
            ("peak_mem_gb", "peak GB", "{:.2f}")]
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for s in rows:
        cells = []
        for key, _, spec in cols:
            v = s.get(key)
            cells.append(spec.format(v) if isinstance(v, (int, float)) else (str(v) if v is not None else "—"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--gpu-index", type=int, default=1, help="nvidia-smi index of the inference GPU (default 1, matches the plotter)")
    p.add_argument("--warmup-clips", type=int, default=2, help="module_fps.csv rows to drop before taking the steady-state median")
    p.add_argument("--no-plot", action="store_true", help="skip rendering gpu_curves.png")
    p.add_argument("--markdown", default=None, help="also write the comparison table to this .md file")
    a = p.parse_args()

    rows = []
    for d in a.run_dirs:
        if not os.path.isdir(d):
            print(f"WARN: not a dir, skipping: {d}", file=sys.stderr)
            continue
        rows.append(summarize_run(d, a.gpu_index, a.warmup_clips))
        if not a.no_plot and os.path.exists(os.path.join(d, "gpu_telemetry.csv")):
            if os.path.exists(PLOTTER):
                r = subprocess.run([sys.executable, PLOTTER, d, str(a.gpu_index)],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    print(f"[plot] {d}/gpu_curves.png", file=sys.stderr)
                else:
                    print(f"[plot FAILED] {d}: {r.stderr.strip()}", file=sys.stderr)
            else:
                print(f"WARN: {PLOTTER} not found (run from the repo root to enable plotting)", file=sys.stderr)

    for s in rows:
        print_run(s)

    if len(rows) >= 1:
        table = markdown_table(rows)
        print("\n" + table)
        if a.markdown:
            with open(a.markdown, "w") as f:
                f.write("# Streaming benchmark comparison\n\n" + table + "\n")
            print(f"\nwrote {a.markdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
