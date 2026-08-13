#!/usr/bin/env python3
"""Concatenate per-clip eval videos into one timeline per run, and optionally
build a left/right side-by-side comparison of several runs.

Each eval run dir (e.g. output/<run>-<step>-sample100/) holds one mp4 per clip,
named by the clip's GT-video basename (and usually each clip is already vertically
stacked: model output over GT). This tool:

  * picks the clip order (from a metadata CSV's `video` column, or sorted filename),
  * keeps only clips present in ALL given runs (so panels stay frame-aligned),
  * concatenates each run's clips in that order -> one video per run,
  * if >1 run, hstacks the runs frame-by-frame -> a comparison video,
  * optionally overlays the clip index + filename in the top-left corner.

Streams one frame at a time (low, flat memory). Must run where imageio + an
ffmpeg backend are available — on this cluster that means inside the singularity
container (use --launch, or run under `singularity exec`).
"""
import argparse, csv, os, sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

MAX_FRAMES = 2000  # per-clip safety cap (mp4 metadata often reports nframes=inf)


def _load_font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def label_frame(frame, text, font):
    """Draw `text` at the top-left corner with a dark backing box for legibility."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    x, y = 8, 6
    l, t, r, b = draw.textbbox((x, y), text, font=font)
    draw.rectangle([l - 4, t - 3, r + 4, b + 3], fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 0), font=font)
    return np.asarray(img)


def clip_order(args, run_dirs):
    """Return an ordered list of clip filenames present in EVERY run dir."""
    if args.order_csv:
        col = args.csv_col
        with open(args.order_csv, newline="") as f:
            names = [os.path.splitext(os.path.basename(r[col].strip()))[0] + ".mp4"
                     for r in csv.DictReader(f)]
    else:
        # sorted union of filenames across runs
        names = sorted({n for d in run_dirs for n in os.listdir(d) if n.endswith(".mp4")})

    ordered, missing = [], []
    for n in names:
        if all(os.path.isfile(os.path.join(d, n)) for d in run_dirs):
            ordered.append(n)
        else:
            missing.append(n)
    return ordered, missing


def resize_h(frame, h):
    if frame.shape[0] == h:
        return frame
    neww = int(round(frame.shape[1] * h / frame.shape[0]))
    return np.asarray(Image.fromarray(frame).resize((neww, h)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="one or more run output dirs (each becomes a left->right panel)")
    ap.add_argument("--labels", default=None,
                    help="comma-separated short names for the runs (default: dir basenames); "
                         "used in output filenames")
    ap.add_argument("--order-csv", dest="order_csv", default=None,
                    help="metadata CSV; clips are ordered by its `video` column (basename). "
                         "Omit to order by sorted filename.")
    ap.add_argument("--csv-col", dest="csv_col", default="video",
                    help="CSV column holding the clip path (default: video)")
    ap.add_argument("--out-dir", dest="out_dir", default="output/compare",
                    help="output directory (default: output/compare)")
    ap.add_argument("--out-prefix", dest="out_prefix", default="compare",
                    help="basename prefix for the comparison video (default: compare)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--label", dest="label", action="store_true", default=True,
                    help="overlay clip index + filename top-left on the comparison (default on)")
    ap.add_argument("--no-label", dest="label", action="store_false")
    ap.add_argument("--no-comparison", dest="comparison", action="store_false", default=True,
                    help="only write per-run concat videos, skip the side-by-side")
    args = ap.parse_args()

    run_dirs = [d.rstrip("/") for d in args.runs]
    for d in run_dirs:
        if not os.path.isdir(d):
            sys.exit(f"run dir not found: {d}")
    names = (args.labels.split(",") if args.labels
             else [os.path.basename(d) for d in run_dirs])
    if len(names) != len(run_dirs):
        sys.exit(f"--labels count ({len(names)}) != runs count ({len(run_dirs)})")

    os.makedirs(args.out_dir, exist_ok=True)
    ordered, missing = clip_order(args, run_dirs)
    print(f"runs: {len(run_dirs)} | clips usable (in all runs): {len(ordered)} | "
          f"skipped (not in all): {len(missing)}")
    for n in missing[:20]:
        print(f"  skip {n}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    if not ordered:
        sys.exit("No clips common to all runs; nothing to do.")

    # Per-run concat writers
    per_run_paths = [os.path.join(args.out_dir, f"{nm}-concat.mp4") for nm in names]
    writers = [imageio.get_writer(p, fps=args.fps, quality=5, macro_block_size=1)
               for p in per_run_paths]
    make_cmp = args.comparison and len(run_dirs) > 1
    cmp_path = os.path.join(args.out_dir, f"{args.out_prefix}_{'_vs_'.join(names)}_lr.mp4")
    wc = (imageio.get_writer(cmp_path, fps=args.fps, quality=5, macro_block_size=1)
          if make_cmp else None)
    font = None

    for i, name in enumerate(ordered):
        readers = [imageio.get_reader(os.path.join(d, name)) for d in run_dirs]
        try:
            for f, frames in enumerate(zip(*readers)):
                if f >= MAX_FRAMES:
                    break
                for w, fr in zip(writers, frames):
                    w.append_data(fr)
                if make_cmp:
                    h = frames[0].shape[0]
                    panels = [resize_h(fr, h) for fr in frames]
                    cmp_frame = np.concatenate(panels, axis=1)
                    if args.label:
                        if font is None:
                            font = _load_font(max(16, cmp_frame.shape[0] // 36))
                        cmp_frame = label_frame(cmp_frame, f"{i+1:03d}  {os.path.splitext(name)[0]}", font)
                    wc.append_data(cmp_frame)
        finally:
            for r in readers:
                r.close()
        if (i + 1) % 10 == 0 or i + 1 == len(ordered):
            print(f"  composited {i+1}/{len(ordered)} clips")

    for w in writers:
        w.close()
    if wc:
        wc.close()
    print("DONE")
    for p in per_run_paths:
        print(f"  {p}")
    if make_cmp:
        print(f"  {cmp_path}")


if __name__ == "__main__":
    main()
