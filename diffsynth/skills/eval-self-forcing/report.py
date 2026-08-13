"""Turn temporal_metrics.py output into a metric matrix (markdown) + drift curves (png).

Usage: python report.py --metrics DIR [--out DIR]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz reference palette, categorical slots 1 & 2 (validated light-mode, all checks pass)
C_A, C_B = "#2a78d6", "#eb6834"
C_GT, INK, INK2, GRID = "#52514e", "#0b0b0b", "#52514e", "#dcdcd8"
SURFACE = "#fcfcfb"
FPS = 30.0

# metric key -> (label, unit, direction) where direction is
#   'gt'   = closeness to the GT value is what matters
#   'down' = lower is better
#   'up'   = higher is better
METRICS = [
    ("psnr",           "PSNR vs GT",                        "dB",   "up"),
    ("tlp",            "Temporal LPIPS  LPIPS(I_t, I_t+1)", "",     "gt"),
    ("tdiff",          "Frame difference  |I_t+1 - I_t|",   "0-255", "gt"),
    ("warp_self",      "Warp error (own flow)",             "0-255", "gt"),
    ("warp_gtflow",    "Warp error (GT flow)",              "0-255", "down"),
    ("tof",            "Flow deviation vs GT (tOF)",        "px",   "down"),
    ("static",         "Static-background flicker",         "0-255", "gt"),
]


def fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def matrix_table(df, region, la, lb):
    lines = [
        f"| Metric | Unit | GT | {la} | {lb} | Better | Δ (SF − base) |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label, unit, direction in METRICS:
        ka, kb, kg = f"{key}_a_{region}", f"{key}_b_{region}", f"{key}_gt_{region}"
        if ka not in df or kb not in df:
            continue
        a, b = df[ka].mean(), df[kb].mean()
        g = df[kg].mean() if kg in df else None
        nd = 2 if key == "psnr" else (4 if key == "tlp" else 3)
        if direction == "up":
            win = la if a > b else lb if b > a else "tie"
        elif direction == "down":
            win = la if a < b else lb if b < a else "tie"
        else:  # closeness to GT
            win = la if abs(a - g) < abs(b - g) else lb if abs(b - g) < abs(a - g) else "tie"
        arrow = {"up": "higher", "down": "lower", "gt": "closer to GT"}[direction]
        lines.append(
            f"| {label} | {unit or '—'} | {fmt(g, nd)} | {fmt(a, nd)} | {fmt(b, nd)} "
            f"| **{win}** ({arrow}) | {b - a:+.{nd}f} |"
        )
    return "\n".join(lines)


def per_clip_table(df, region, la, lb):
    cols = ["clip", "frames",
            f"psnr_a_{region}", f"psnr_b_{region}",
            f"warp_self_gt_{region}", f"warp_self_a_{region}", f"warp_self_b_{region}",
            f"tof_a_{region}", f"tof_b_{region}"]
    cols = [c for c in cols if c in df]
    hdr = ["clip", "n", f"PSNR {la}", f"PSNR {lb}", "warp GT", f"warp {la}", f"warp {lb}",
           f"tOF {la}", f"tOF {lb}"][:len(cols)]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(str(v) if c in ("clip", "frames") else f"{v:.2f}"
                         if "psnr" in c else f"{v:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mean_series(npz, clips, key, nmin=None):
    arrs = [npz[f"{c}|{key}"] for c in clips if f"{c}|{key}" in npz]
    if not arrs:
        return None
    n = nmin or min(len(a) for a in arrs)
    return np.mean([a[:n] for a in arrs], axis=0)


def smooth(y, w=15):
    if y is None or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_xlabel("time in clip (s)", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_facecolor(SURFACE)


def end_labels(ax, items, min_gap_frac=0.055):
    """Direct-label each line at its right end, nudged apart so they never collide."""
    lo, hi = ax.get_ylim()
    span = hi - lo or 1.0
    items = sorted(items, key=lambda it: it[1])  # by y, bottom-up
    placed = []
    for x_end, y_end, text, color in items:
        y = y_end
        if placed and (y - placed[-1]) < min_gap_frac * span:
            y = placed[-1] + min_gap_frac * span
        placed.append(y)
        ax.annotate(text, (x_end, y), xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=9, fontweight="bold", va="center",
                    annotation_clip=False)


def curves(df, npz, clips, la, lb, out_png):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor=SURFACE)
    fig.suptitle("Temporal behaviour over the 10 s rollout — mean over "
                 f"{len(clips)} clips (15-frame moving average)",
                 color=INK, fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.985)

    panels = [
        (axes[0][0], "psnr", "full", "Reconstruction quality vs GT — full frame",
         "PSNR (dB)  ↑ better", False),
        (axes[0][1], "psnr", "head", "Reconstruction quality vs GT — head crop",
         "PSNR (dB)  ↑ better", False),
        (axes[1][0], "warp_self", "full", "Temporal warp error (own flow) — full frame",
         "|I_t+1 − W(I_t)|  (0-255)", True),
        (axes[1][1], "warp_self", "head", "Temporal warp error (own flow) — head crop",
         "|I_t+1 − W(I_t)|  (0-255)", True),
    ]
    for ax, key, region, title, ylabel, show_gt in panels:
        series = {}
        if show_gt:
            series["Ground truth"] = (mean_series(npz, clips, f"{key}_gt_{region}"), C_GT, "--")
        series[la] = (mean_series(npz, clips, f"{key}_a_{region}"), C_A, "-")
        series[lb] = (mean_series(npz, clips, f"{key}_b_{region}"), C_B, "-")
        tips, bodies = [], []
        for name, (y, color, ls) in series.items():
            if y is None:
                continue
            y = smooth(y)
            x = np.arange(len(y)) / FPS
            ax.plot(x, y, color=color, linewidth=2.0, linestyle=ls, label=name,
                    solid_capstyle="round")
            tips.append((x[-1], y[-1], name, color))
            bodies.append(y[max(int(0.3 * FPS), 1):])  # skip the ref->frame-0 transition
        style(ax, title, ylabel)
        ax.margins(x=0.14)
        if bodies:
            # keep the frame-0 transition spike out of the y-scale so the rollout is readable
            flat = np.concatenate(bodies)
            pad = 0.08 * (flat.max() - flat.min() + 1e-6)
            ax.set_ylim(flat.min() - pad, flat.max() + pad)
        end_labels(ax, tips)
        leg = ax.legend(frameon=False, fontsize=9, loc="best", labelcolor=INK2)
        for t in leg.get_texts():
            t.set_color(INK2)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def clip_bars(df, la, lb, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), facecolor=SURFACE, sharey=True)
    y = np.arange(len(df))
    short = [c[:2] + " " + c[3:13] for c in df["clip"]]
    for ax, region, title in ((axes[0], "full", "Per-clip PSNR vs GT — full frame"),
                              (axes[1], "head", "Per-clip PSNR vs GT — head crop")):
        h = 0.38
        ax.barh(y - h / 2, df[f"psnr_a_{region}"], height=h - 0.03, color=C_A, label=la)
        ax.barh(y + h / 2, df[f"psnr_b_{region}"], height=h - 0.03, color=C_B, label=lb)
        ax.set_yticks(y, short, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=10)
        ax.set_xlabel("PSNR (dB)  ↑ better", color=INK2, fontsize=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.set_facecolor(SURFACE)
        leg = ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)
        for t in leg.get_texts():
            t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="dir written by temporal_metrics.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    md = Path(args.metrics)
    out = Path(args.out or md)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(md / "temporal_metrics_per_clip.csv").sort_values("clip").reset_index(drop=True)
    labels = json.loads((md / "labels.json").read_text())
    la, lb = labels["a"], labels["b"]
    npz = np.load(md / "temporal_series.npz")
    clips = list(df["clip"])

    curves(df, npz, clips, la, lb, out / "temporal_curves.png")
    clip_bars(df, la, lb, out / "per_clip_psnr.png")

    doc = [
        "# Temporal-consistency matrix",
        "",
        f"- **A = {la}**   **B = {lb}**",
        f"- {len(df)} clips, {int(df['frames'].sum())} frames total "
        f"({df['frames'].iloc[0]:.0f} frames = {df['frames'].iloc[0]/FPS:.1f} s per clip @ {FPS:.0f} fps)",
        f"- Full-frame metrics computed at 640x360; head-crop metrics at native "
        f"{384}x{384}.  Static-pixel fraction: "
        f"{df['static_px_frac_full'].mean()*100:.1f}% (full), "
        f"{df['static_px_frac_head'].mean()*100:.1f}% (head).",
        "",
        "For **frame difference, warp error (own flow), temporal LPIPS and static flicker the "
        "GT column is the target, not a floor** — a value far *below* GT means the model is "
        "over-smoothed/frozen, far *above* means it flickers.",
        "",
        "## Full frame",
        "",
        matrix_table(df, "full", la, lb),
        "",
        "## Head crop (384x384, face-centred)",
        "",
        matrix_table(df, "head", la, lb),
        "",
        "## Per clip (full frame)",
        "",
        per_clip_table(df, "full", la, lb),
        "",
        "## Metric definitions",
        "",
        "| Metric | Definition |",
        "|---|---|",
        "| PSNR vs GT | `10 log10(255² / MSE(gen, gt))`, per frame then averaged. |",
        "| Temporal LPIPS | `LPIPS_alex(I_t, I_t+1)` averaged over t — perceptual size of the "
        "frame-to-frame change. |",
        "| Frame difference | `mean|I_t+1 − I_t|` — raw flicker energy. |",
        "| Warp error (own flow) | Farnebäck flow `f` computed on the video itself, "
        "`mean|I_t+1 − W(I_t; f)|` over pixels passing a 1.5 px forward-backward "
        "consistency check. Pure self-consistency, GT-independent. |",
        "| Warp error (GT flow) | Same, but warping with the **GT's** flow field — punishes "
        "content that does not move the way GT moves. Lower is better. |",
        "| Flow deviation (tOF) | `mean‖f_gen − f_gt‖₂` in pixels. |",
        "| Static-background flicker | `mean|I_t+1 − I_t|` restricted to pixels whose GT "
        "temporal std < 2/255 — measures background boiling. |",
        "",
        "## Figures",
        "",
        "- `temporal_curves.png` — PSNR and warp error across the 10 s rollout (drift check).",
        "- `per_clip_psnr.png` — per-clip PSNR, full frame and head crop.",
    ]
    (out / "temporal_consistency_matrix.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc[:40]))
    print(f"\n[info] wrote {out/'temporal_consistency_matrix.md'}, "
          f"{out/'temporal_curves.png'}, {out/'per_clip_psnr.png'}")


if __name__ == "__main__":
    main()
