#!/usr/bin/env python3
"""Make raw field recordings usable as voice-clone references.

Field audio is typically too quiet and mostly silence; feeding it straight to a
cloning model produces quiet, runaway takes. This drops near-silent frames and
normalizes loudness, which is what separates "the model is bad" from "the input is bad".

  python prep_refs.py --list infer_list_audio.txt --root /path/to/repo --out refs_clean/
  python prep_refs.py ref1.wav ref2.wav --out refs_clean/

Writes <out>/<id>.wav plus <out>/../refs_clean.json (id, path, transcript slot).
Run it on your own references before comparing engines, and keep the raw set too:
the raw-vs-clean delta is the informative number.
"""
import argparse, json, os, sys
import numpy as np, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("wavs", nargs="*", help="reference wavs (or use --list)")
ap.add_argument("--list", help="text file with one audio path per line")
ap.add_argument("--root", default="", help="prefix for relative paths in --list")
ap.add_argument("--out", required=True, help="output directory")
ap.add_argument("--target-rms", type=float, default=0.08)
ap.add_argument("--gate-dbfs", type=float, default=-45.0,
                help="20ms frames quieter than this are treated as silence")
ap.add_argument("--pad-ms", type=float, default=60.0, help="keep this much around each kept run")
ap.add_argument("--transcripts", help="optional json {basename: text} to carry into the index")
a = ap.parse_args()

paths = list(a.wavs)
if a.list:
    paths += [os.path.join(a.root, l.strip()) for l in open(a.list) if l.strip()]
if not paths:
    sys.exit("no input audio")
os.makedirs(a.out, exist_ok=True)
tr = json.load(open(a.transcripts)) if a.transcripts else {}

print(f"{'id':<8}{'raw dur':>9}{'clean dur':>11}{'raw rms':>9}{'clean rms':>11}  file")
rows = []
for i, p in enumerate(paths, 1):
    w, sr = sf.read(p)
    m = w.mean(1) if w.ndim > 1 else w
    fr = int(0.02 * sr); n = len(m) // fr
    if n == 0:
        sys.exit(f"{p}: shorter than one 20ms frame")
    fs = m[: n * fr].reshape(n, fr)
    keep = np.sqrt((fs ** 2).mean(1)) > 10 ** (a.gate_dbfs / 20)
    pad = max(1, int(a.pad_ms / 20))
    k = keep.copy()
    for j in np.where(keep)[0]:
        k[max(0, j - pad): min(n, j + pad + 1)] = True
    if k.sum() < n * 0.05:            # never gut a clip entirely
        k[:] = True
    v = fs[k].reshape(-1)
    v = np.clip(v / max(np.sqrt((v ** 2).mean()), 1e-9) * a.target_rms, -0.99, 0.99)
    rid = f"ref{i:02d}"
    outp = os.path.join(a.out, f"{rid}.wav")
    sf.write(outp, v, sr)
    base = os.path.basename(p)
    rows.append({"id": rid, "path": os.path.abspath(outp), "orig_path": os.path.abspath(p),
                 "file": base, "text": tr.get(base, {}).get("text", "") if tr else ""})
    print(f"{rid:<8}{len(m)/sr:8.2f}s{len(v)/sr:10.2f}s"
          f"{np.sqrt((m**2).mean()):9.3f}{np.sqrt((v**2).mean()):11.3f}  {base}")
    if len(v) / sr < 3.0:
        print(f"         ^ only {len(v)/sr:.1f}s of speech survived -- weak reference")

idx = os.path.join(os.path.dirname(os.path.abspath(a.out)), "refs_clean.json")
json.dump(rows, open(idx, "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {len(rows)} refs + {idx}")
