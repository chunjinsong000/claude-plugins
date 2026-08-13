"""Temporal-consistency metrics for two eval runs against their shared ground truth.

Each eval clip is a 2560x720 mp4 laid out `left_gt_right_gen`.  For every clip the
script streams GT, run-A generation and run-B generation together and computes:

  psnr            PSNR of the generation vs GT (dB, higher better) - not temporal,
                  reported per second as well so error accumulation / drift is visible.
  tdiff           mean |I_t+1 - I_t| in 0-255 (raw flicker; GT is the reference value,
                  much lower than GT = over-smooth, much higher = flickery)
  warp_self       warping error using the video's OWN optical flow: mean
                  |I_t+1 - W(I_t; f_self)| over non-occluded pixels. Pure temporal
                  smoothness, independent of GT.
  warp_gtflow     warping error using the GT's optical flow. Penalises content that
                  fails to move the way GT moves (motion fidelity + consistency).
  tof             mean ||f_gen - f_gt||_2 in pixels: how far the generated motion
                  field is from the GT motion field.
  tlp             mean LPIPS(I_t, I_t+1) - perceptual temporal change.  Compared with
                  GT's value as |tlp - tlp_gt| ("dtlp").
  static_flicker  mean |I_t+1 - I_t| restricted to pixels that are static in GT
                  (background shimmer / boiling).

Every metric is computed full-frame and again inside the per-clip head box, since the
face is where causal drift shows up first.

Usage:
  python temporal_metrics.py --a DIR --b DIR --out DIR [--head-boxes JSON]
                             [--label-a STR] [--label-b STR] [--jobs N] [--no-lpips]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PANEL_W, PANEL_H = 1280, 720
FLOW_W, FLOW_H = 640, 360          # flow / warp resolution (full-frame metrics)
OCC_THRESH = 1.5                   # px, fwd-bwd flow consistency
STATIC_THRESH = 2.0                # 0-255, GT temporal std below this == static pixel
METHODS = ("gt", "a", "b")

cv2.setNumThreads(1)               # we parallelise over clips instead

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)


def flow(g0, g1):
    return cv2.calcOpticalFlowFarneback(g0, g1, None, **FARNEBACK)


def warp(img, f):
    """Warp img (frame t) forward by flow f (t -> t+1) so it aligns with frame t+1."""
    h, w = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx - f[..., 0], gy - f[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def occlusion_mask(f_fwd, f_bwd):
    """Valid (non-occluded) pixels via forward-backward flow consistency."""
    fb = warp(f_bwd, f_fwd)
    err = np.linalg.norm(f_fwd + fb, axis=-1)
    return (err < OCC_THRESH).astype(np.float32)


def masked_mean(err, mask):
    s = mask.sum()
    return float((err * mask).sum() / s) if s > 0 else float("nan")


class Acc:
    """Accumulates the per-frame series for one video at one scale."""

    def __init__(self):
        self.psnr, self.tdiff, self.warp_self, self.warp_gtflow, self.tof = ([] for _ in range(5))
        self.static = []


def region_metrics(prev, cur, gt_prev, gt_cur, gflow_fwd, valid, static_mask, acc, is_gt):
    """prev/cur: BGR float32 0-255 frames of the video under test (already at metric res)."""
    # PSNR vs GT (0 for the GT itself -> skipped by the caller)
    if not is_gt:
        mse = float(np.mean((cur - gt_cur) ** 2))
        acc.psnr.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-8)))

    acc.tdiff.append(float(np.mean(np.abs(cur - prev))))
    if static_mask is not None and static_mask.sum() > 0:
        d = np.abs(cur - prev).mean(axis=-1)
        acc.static.append(masked_mean(d, static_mask))

    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    f_self = gflow_fwd if is_gt else flow(g0, g1)
    acc.warp_self.append(masked_mean(np.abs(warp(prev, f_self) - cur).mean(axis=-1), valid))
    acc.warp_gtflow.append(masked_mean(np.abs(warp(prev, gflow_fwd) - cur).mean(axis=-1), valid))
    if not is_gt:
        acc.tof.append(float(np.mean(np.linalg.norm(f_self - gflow_fwd, axis=-1))))


def static_mask_for(path, box, n=40):
    """Pixels whose GT value barely changes over the clip (background)."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    full, head = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        gt = fr[:PANEL_H, :PANEL_W]
        full.append(cv2.cvtColor(cv2.resize(gt, (FLOW_W, FLOW_H)), cv2.COLOR_BGR2GRAY))
        x, y, w, h = box
        head.append(cv2.cvtColor(gt[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))
    cap.release()
    fm = (np.std(np.array(full, np.float32), axis=0) < STATIC_THRESH).astype(np.float32)
    hm = (np.std(np.array(head, np.float32), axis=0) < STATIC_THRESH).astype(np.float32)
    return fm, hm


def process_clip(job):
    stem, a_mp4, b_mp4, box, stride = job
    x, y, w, h = box
    sm_full, sm_head = static_mask_for(a_mp4, box)

    ca, cb = cv2.VideoCapture(str(a_mp4)), cv2.VideoCapture(str(b_mp4))
    accs = {(m, r): Acc() for m in METHODS for r in ("full", "head")}
    prev = None
    n = 0
    while True:
        oka, fa = ca.read()
        okb, fb = cb.read()
        if not (oka and okb):
            break
        panels = {
            "gt": fa[:PANEL_H, :PANEL_W],
            "a": fa[:PANEL_H, PANEL_W:PANEL_W * 2],
            "b": fb[:PANEL_H, PANEL_W:PANEL_W * 2],
        }
        cur = {}
        for m, p in panels.items():
            cur[(m, "full")] = cv2.resize(p, (FLOW_W, FLOW_H)).astype(np.float32)
            cur[(m, "head")] = p[y:y + h, x:x + w].astype(np.float32)
        if prev is not None and n % stride == 0:
            for region, smask in (("full", sm_full), ("head", sm_head)):
                g0 = cv2.cvtColor(prev[("gt", region)], cv2.COLOR_BGR2GRAY)
                g1 = cv2.cvtColor(cur[("gt", region)], cv2.COLOR_BGR2GRAY)
                f_fwd = flow(g0, g1)
                valid = occlusion_mask(f_fwd, flow(g1, g0))
                for m in METHODS:
                    region_metrics(prev[(m, region)], cur[(m, region)],
                                   prev[("gt", region)], cur[("gt", region)],
                                   f_fwd, valid, smask, accs[(m, region)], m == "gt")
        prev = cur
        n += 1
    ca.release()
    cb.release()

    row = {"clip": stem, "frames": n}
    series = {}
    for (m, region), acc in accs.items():
        for name in ("psnr", "tdiff", "warp_self", "warp_gtflow", "tof", "static"):
            v = getattr(acc, name)
            if not v:
                continue
            row[f"{name}_{m}_{region}"] = float(np.mean(v))
            series[f"{name}_{m}_{region}"] = np.array(v, np.float32)
    row["static_px_frac_full"] = float(sm_full.mean())
    row["static_px_frac_head"] = float(sm_head.mean())
    return row, series


def add_lpips(rows, jobs, boxes, device="cuda"):
    """Streaming temporal LPIPS: keeps only the previous frame per (method, region)."""
    import torch
    import lpips
    net = lpips.LPIPS(net="alex").to(device).eval()

    def to_t(img):
        t = torch.from_numpy(img[..., ::-1].copy()).permute(2, 0, 1)[None]
        return t.float().div(127.5).sub(1.0).to(device)

    for row, (stem, a_mp4, b_mp4, box, *_) in zip(rows, jobs):
        x, y, w, h = box
        ca, cb = cv2.VideoCapture(str(a_mp4)), cv2.VideoCapture(str(b_mp4))
        prev, vals = {}, {k: [] for k in ((m, r) for m in METHODS for r in ("full", "head"))}
        with torch.no_grad():
            while True:
                oka, fa = ca.read()
                okb, fb = cb.read()
                if not (oka and okb):
                    break
                panels = {"gt": fa[:PANEL_H, :PANEL_W],
                          "a": fa[:PANEL_H, PANEL_W:PANEL_W * 2],
                          "b": fb[:PANEL_H, PANEL_W:PANEL_W * 2]}
                for m, p in panels.items():
                    for region, img in (("full", cv2.resize(p, (FLOW_W, FLOW_H))),
                                        ("head", p[y:y + h, x:x + w])):
                        t = to_t(img)
                        if (m, region) in prev:
                            vals[(m, region)].append(float(net(prev[(m, region)], t).item()))
                        prev[(m, region)] = t
        ca.release()
        cb.release()
        for (m, region), v in vals.items():
            row[f"tlp_{m}_{region}"] = float(np.mean(v)) if v else float("nan")
        for region in ("full", "head"):
            for m in ("a", "b"):
                row[f"dtlp_{m}_{region}"] = abs(row[f"tlp_{m}_{region}"] - row[f"tlp_gt_{region}"])
        print(f"  lpips {row['clip']}: gt={row['tlp_gt_full']:.4f} "
              f"a={row['tlp_a_full']:.4f} b={row['tlp_b_full']:.4f}")
    del net
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-boxes", default=None)
    ap.add_argument("--label-a", default="Baseline")
    ap.add_argument("--label-b", default="Self-Forcing")
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--stride", type=int, default=1,
                    help="measure every Nth consecutive frame pair (flow metrics only); "
                         "2 halves the runtime and only thins the drift curves")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="only these clips; matched as a prefix of the clip stem")
    args = ap.parse_args()

    a_dir, b_dir, out_dir = Path(args.a), Path(args.b), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def stems(d):
        return {p.stem for p in d.glob("*.mp4") if not p.stem.startswith("concat")}

    common = sorted(stems(a_dir) & stems(b_dir))
    if args.clips:
        common = [s for s in common if any(s.startswith(p) for p in args.clips)]
        if not common:
            raise SystemExit(f"--clips {args.clips} matched none of the shared clips")
    boxes = json.load(open(args.head_boxes)) if args.head_boxes else {}
    default_box = [int(PANEL_W * 0.54 - 192), 0, 384, 384]
    jobs = [(s, a_dir / f"{s}.mp4", b_dir / f"{s}.mp4", boxes.get(s, default_box), args.stride)
            for s in common]
    print(f"[info] {len(jobs)} clip(s) on {args.jobs} worker(s)")

    rows, all_series = [], {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for row, series in ex.map(process_clip, jobs):
            rows.append(row)
            all_series[row["clip"]] = series
            print(f"  flow  {row['clip']}: frames={row['frames']} "
                  f"psnr a={row['psnr_a_full']:.2f} b={row['psnr_b_full']:.2f}")
    rows.sort(key=lambda r: r["clip"])
    jobs.sort(key=lambda j: j[0])

    if not args.no_lpips:
        add_lpips(rows, jobs, boxes)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "temporal_metrics_per_clip.csv", index=False)
    np.savez_compressed(out_dir / "temporal_series.npz",
                        **{f"{c}|{k}": v for c, s in all_series.items() for k, v in s.items()})
    (out_dir / "labels.json").write_text(json.dumps(
        {"a": args.label_a, "b": args.label_b, "clips": common}, indent=2))
    print(f"\n[info] wrote {out_dir/'temporal_metrics_per_clip.csv'}")
    print(df[[c for c in df.columns if c.endswith("_full") or c in ("clip", "frames")]].to_string())


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    main()
