"""Detect a per-clip head crop box on the GT (left) panel of the eval videos.

The GT and generated panels are pose-aligned, so one box is reused for every panel.
Writes {clip_stem: [x, y, w, h]} as JSON.

Usage: python detect_heads.py <clip_dir> <out.json> [crop_side]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PANEL_W, PANEL_H = 1280, 720
N_SAMPLE = 24

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")


def detect_clip(path: Path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n - 1, 0), N_SAMPLE).astype(int)
    centers, sizes = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame[:PANEL_H, :PANEL_W], cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            faces = profile.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        centers.append((x + w / 2, y + h / 2))
        sizes.append((w + h) / 2)
    cap.release()
    if not centers:
        return None, 0.0
    return np.median(np.array(centers), axis=0), float(np.median(sizes))


def main(clip_dir, out_json, crop=384):
    crop = int(crop)
    boxes = {}
    for mp4 in sorted(Path(clip_dir).glob("*.mp4")):
        if mp4.stem.startswith("concat"):
            continue
        c, size = detect_clip(mp4)
        if c is None:
            cx, cy = PANEL_W * 0.54, PANEL_H * 0.27
            print(f"{mp4.stem}: NO FACE -> fallback centre")
        else:
            cx, cy = c
            cy -= 0.10 * crop  # bias up so hair / top of head stays inside
        x = int(round(np.clip(cx - crop / 2, 0, PANEL_W - crop)))
        y = int(round(np.clip(cy - crop / 2, 0, PANEL_H - crop)))
        boxes[mp4.stem] = [x, y, crop, crop]
        print(f"{mp4.stem}: face={size:.0f}px box=({x},{y},{crop},{crop})")
    Path(out_json).write_text(json.dumps(boxes, indent=2))
    print(f"\nwrote {out_json} ({len(boxes)} clips)")


if __name__ == "__main__":
    main(*sys.argv[1:])
