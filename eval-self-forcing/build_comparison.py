"""Build GT | baseline | self-forcing comparison videos from two eval run dirs.

Each eval clip is a 2560x720 mp4 laid out `left_gt_right_gen` (left half = GT,
right half = generation).  This script emits, per clip and then concatenated:

  * full   -> 3 panels at native 1280x720  (GT | A | B)   = 3840x720
  * head   -> the same 3 panels cropped to a per-clip head box and 2x upscaled

Only clips present in *both* run dirs are used, so the panels stay frame-aligned.

Usage:
  python build_comparison.py --a DIR --b DIR --out DIR
                             [--label-a STR] [--label-b STR]
                             [--head-boxes JSON] [--jobs N] [--crf N]
                             [--skip-full] [--skip-head]
"""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PANEL_W, PANEL_H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def esc(text: str) -> str:
    """Escape a string for ffmpeg drawtext."""
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def label(text, fontsize, x="(w-text_w)/2", y="8"):
    return (
        f"drawtext=fontfile={FONT}:text='{esc(text)}':fontcolor=white:fontsize={fontsize}"
        f":box=1:boxcolor=black@0.65:boxborderw=10:x={x}:y={y}"
    )


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n{p.stdout[-4000:]}")
    return p.stdout


def clip_cmd(a_mp4, b_mp4, out, idx, stem, labels, box, crf, head):
    """One ffmpeg invocation producing a 3-panel clip (full-frame or head-zoom)."""
    if head:
        x, y, w, h = box
        sc = 2 if w * 2 <= 900 else 1
        crop = f"crop={w}:{h}:{{ox}}:{y},scale={w*sc}:{h*sc}:flags=lanczos"
        fs, hdr_fs = 30, 26
    else:
        crop = f"crop={PANEL_W}:{PANEL_H}:{{ox}}:0"
        fs, hdr_fs = 40, 34

    # GT + generation A from file 0; generation B from file 1
    fc = [
        f"[0:v]{crop.format(ox=box[0] if head else 0)},{label(labels[0], fs)}[p0]",
        f"[0:v]{crop.format(ox=(box[0] + PANEL_W) if head else PANEL_W)},{label(labels[1], fs)}[p1]",
        f"[1:v]{crop.format(ox=(box[0] + PANEL_W) if head else PANEL_W)},{label(labels[2], fs)}[p2]",
        f"[p0][p1][p2]hstack=inputs=3[st]",
        f"[st]{label(f'{idx:02d}  {stem}', hdr_fs, x='12', y='h-th-12')}[v]",
    ]
    return [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(a_mp4), "-i", str(b_mp4),
        "-filter_complex", ";".join(fc),
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(out),
    ]


def concat(parts, out):
    lst = out.parent / (out.stem + "_list.txt")
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
         "-c", "copy", "-movflags", "+faststart", str(out)])
    lst.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run A clip dir (the baseline)")
    ap.add_argument("--b", required=True, help="run B clip dir (self-forcing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-a", default="Baseline")
    ap.add_argument("--label-b", default="Self-Forcing")
    ap.add_argument("--label-gt", default="Ground Truth")
    ap.add_argument("--head-boxes", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-head", action="store_true")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="only these clips; a clip is matched if any value is a prefix of "
                         "its stem, so '00' selects 00_2025-... ")
    args = ap.parse_args()

    a_dir, b_dir, out_dir = Path(args.a), Path(args.b), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_clips"
    tmp.mkdir(exist_ok=True)

    def stems(d):
        return {p.stem for p in d.glob("*.mp4") if not p.stem.startswith("concat")}

    common = sorted(stems(a_dir) & stems(b_dir))
    only_a, only_b = sorted(stems(a_dir) - set(common)), sorted(stems(b_dir) - set(common))
    if only_a:
        print(f"[warn] {len(only_a)} clip(s) only in A, skipped: {only_a}")
    if only_b:
        print(f"[warn] {len(only_b)} clip(s) only in B, skipped: {only_b}")
    if not common:
        sys.exit("no clips shared between the two dirs")
    if args.clips:
        common = [s for s in common if any(s.startswith(p) for p in args.clips)]
        if not common:
            sys.exit(f"--clips {args.clips} matched none of the shared clips")
    print(f"[info] {len(common)} shared clip(s)")

    boxes = json.load(open(args.head_boxes)) if args.head_boxes else {}
    labels = (args.label_gt, args.label_a, args.label_b)

    jobs, parts_full, parts_head = [], [], []
    for i, stem in enumerate(common):
        a_mp4, b_mp4 = a_dir / f"{stem}.mp4", b_dir / f"{stem}.mp4"
        if not args.skip_full:
            o = tmp / f"full_{i:02d}.mp4"
            parts_full.append(o)
            jobs.append(clip_cmd(a_mp4, b_mp4, o, i, stem, labels, [0, 0, 0, 0], args.crf, False))
        if not args.skip_head:
            box = boxes.get(stem)
            if box is None:
                print(f"[warn] no head box for {stem}, using frame centre")
                box = [int(PANEL_W * 0.54 - 192), 0, 384, 384]
            o = tmp / f"head_{i:02d}.mp4"
            parts_head.append(o)
            jobs.append(clip_cmd(a_mp4, b_mp4, o, i, stem, labels, box, args.crf, True))

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for _ in ex.map(run, jobs):
            pass
    print(f"[info] rendered {len(jobs)} clip panel video(s)")

    if parts_full:
        concat(parts_full, out_dir / "compare_full.mp4")
        print(f"  -> {out_dir/'compare_full.mp4'}")
    if parts_head:
        concat(parts_head, out_dir / "compare_head.mp4")
        print(f"  -> {out_dir/'compare_head.mp4'}")

    # keep per-clip files, they are handy for spot checks
    for p in sorted(tmp.glob("*.mp4")):
        stem_i = int(p.stem.split("_")[1])
        kind = p.stem.split("_")[0]
        p.rename(out_dir / f"{kind}_{stem_i:02d}_{common[stem_i]}.mp4")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
