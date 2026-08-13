#!/usr/bin/env python3
"""
Skill wrapper: compare two card-eval result folders -> HTML summary.

Derives the label / generated-video / ground-truth paths from the two result
folders and invokes make_viz_html.py (the visualization engine in the project
root). Uses only the standard library; the engine itself runs under the project
.venv (cv2 / numpy / matplotlib / scipy).

A "result folder" is an inference_results/<checkpoint> dir produced by
run_pipeline.sh, containing a labels/ subdir of per-frame YOLO label files
named <video>_<frameid>.txt.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_GT_VIDEO = "/leonardo_work/AIFAC_F02_378/shared/livedealer/video_cut"
DEFAULT_PRED_VIDEO_ROOT = "/leonardo_work/AIFAC_F02_378/wxu/DiffSynth/output"


def find_project_root(explicit=None):
    """Locate the card_eval project (dir containing make_viz_html.py).

    The skill may live outside the project (e.g. a shared ~/.claude/skills repo),
    so don't assume a fixed path relative to this file. Search, in order:
    an explicit --project-root, the cwd and its parents, then near this script."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [Path.cwd(), *Path.cwd().parents]
    candidates += list(Path(__file__).resolve().parents)
    for base in candidates:
        if (base / "make_viz_html.py").exists():
            return base
    return None


def video_list_from_labels(labels_dir: Path, out_file: str) -> int:
    """Write '<stem>.mp4' lines for every unique video in a labels dir."""
    stems = {p.stem.rsplit("_", 1)[0] for p in labels_dir.glob("*.txt")}
    with open(out_file, "w") as f:
        for s in sorted(stems):
            f.write(s + ".mp4\n")
    return len(stems)


def main():
    ap = argparse.ArgumentParser(
        description="Compare two card-eval result folders -> HTML summary")
    ap.add_argument("result_a", help="eval result folder A (contains labels/)")
    ap.add_argument("result_b", help="eval result folder B (contains labels/)")
    ap.add_argument("--gt-labels", default=None,
                    help="GT labels dir (default: <result_a parent>/gt/labels)")
    ap.add_argument("--gt-video-dir", default=DEFAULT_GT_VIDEO)
    ap.add_argument("--pred-video-root", default=DEFAULT_PRED_VIDEO_ROOT,
                    help="dir holding generated videos, one subdir per checkpoint name")
    ap.add_argument("--video-a", default=None, help="override generated-video dir for A")
    ap.add_argument("--video-b", default=None, help="override generated-video dir for B")
    ap.add_argument("--name-a", default=None, help="display name for A (default: folder name)")
    ap.add_argument("--name-b", default=None, help="display name for B (default: folder name)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <result_a parent>/compare_<A>__vs__<B>)")
    ap.add_argument("--frame-id", type=int, default=0)
    ap.add_argument("--pose-thresh", type=float, default=0.02)
    ap.add_argument("--pred-panel-height", type=int, default=720)
    ap.add_argument("--pred-width", type=int, default=1280)
    ap.add_argument("--pred-height", type=int, default=720)
    ap.add_argument("--python", default=None,
                    help="python used to run the engine (default: project .venv)")
    ap.add_argument("--project-root", default=None,
                    help="card_eval project dir (default: found from cwd via make_viz_html.py)")
    args = ap.parse_args()

    root = find_project_root(args.project_root)
    if root is None:
        sys.exit("[error] could not find make_viz_html.py — run from the card_eval "
                 "project, or pass --project-root PATH")

    ra, rb = Path(args.result_a).resolve(), Path(args.result_b).resolve()
    name_a = args.name_a or ra.name
    name_b = args.name_b or rb.name
    labels_a, labels_b = ra / "labels", rb / "labels"
    for d in (labels_a, labels_b):
        if not d.is_dir():
            sys.exit(f"[error] missing labels dir: {d}")

    video_a = Path(args.video_a) if args.video_a else Path(args.pred_video_root) / ra.name
    video_b = Path(args.video_b) if args.video_b else Path(args.pred_video_root) / rb.name
    gt_labels = Path(args.gt_labels) if args.gt_labels else ra.parent / "gt" / "labels"
    if not gt_labels.is_dir():
        sys.exit(f"[error] GT labels not found: {gt_labels} (pass --gt-labels)")
    out_dir = (Path(args.out_dir) if args.out_dir
               else ra.parent / f"compare_{name_a}__vs__{name_b}")

    engine = root / "make_viz_html.py"
    if not engine.exists():
        sys.exit(f"[error] engine not found: {engine}")
    venv_py = root / ".venv" / "bin" / "python"
    py = args.python or (str(venv_py) if venv_py.exists() else sys.executable)

    tmp = tempfile.NamedTemporaryFile("w", suffix="_vlist.txt", delete=False)
    tmp.close()
    n = video_list_from_labels(labels_a, tmp.name)

    cmd = [py, str(engine),
           "--gt_labels_dir", str(gt_labels),
           "--gt_video_dir", args.gt_video_dir,
           "--pred_a_labels_dir", str(labels_a),
           "--pred_a_video_dir", str(video_a),
           "--pred_a_name", name_a,
           "--pred_b_labels_dir", str(labels_b),
           "--pred_b_video_dir", str(video_b),
           "--pred_b_name", name_b,
           "--video_list", tmp.name,
           "--out_dir", str(out_dir),
           "--frame_id", str(args.frame_id),
           "--pose_thresh", str(args.pose_thresh),
           "--pred_panel_height", str(args.pred_panel_height),
           "--pred_width", str(args.pred_width),
           "--pred_height", str(args.pred_height)]

    print(f"[compare] A={name_a}  B={name_b}  ({n} clips from A's labels)")
    print(f"[compare] videos: A={video_a}  B={video_b}")
    print(f"[compare] gt_labels={gt_labels}")
    print(f"[compare] out={out_dir}")
    try:
        rc = subprocess.run(cmd).returncode
    finally:
        os.unlink(tmp.name)
    if rc == 0:
        print(f"[compare] DONE -> {out_dir / 'index.html'}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
