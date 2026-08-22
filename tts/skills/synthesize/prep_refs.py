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
ap.add_argument("--keep-names", action="store_true",
                help="name outputs after the input file instead of ref0001.wav; keeps the "
                     "corpus traceable back to the source clip")
ap.add_argument("--jobs", type=int, default=1,
                help="parallel workers; thousands of files are IO+decode bound, so this "
                     "is the difference between minutes and half an hour")
ap.add_argument("--subtype", default="PCM_16",
                help="output encoding; PCM_16 is a 4x size cut over float64 and loses "
                     "nothing a TTS reference needs")
a = ap.parse_args()

paths = list(a.wavs)
if a.list:
    paths += [os.path.join(a.root, l.strip()) for l in open(a.list) if l.strip()]
if not paths:
    sys.exit("no input audio")
os.makedirs(a.out, exist_ok=True)
tr = json.load(open(a.transcripts)) if a.transcripts else {}

def process(job):
    i, p = job
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
    base = os.path.basename(p)
    rid = os.path.splitext(base)[0] if a.keep_names else f"ref{i:05d}"
    outp = os.path.join(a.out, f"{rid}.wav")
    sf.write(outp, v, sr, subtype=a.subtype)
    return {"id": rid, "path": os.path.abspath(outp), "orig_path": os.path.abspath(p),
            "file": base, "text": tr.get(base, {}).get("text", "") if tr else "",
            "raw_seconds": round(len(m) / sr, 2), "seconds": round(len(v) / sr, 2),
            "raw_rms": round(float(np.sqrt((m ** 2).mean())), 4),
            "rms": round(float(np.sqrt((v ** 2).mean())), 4)}


jobs = list(enumerate(paths, 1))
if a.jobs > 1:
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        rows = pool.map(process, jobs, chunksize=16)
else:
    rows = [process(j) for j in jobs]

weak = [r for r in rows if r["seconds"] < 3.0]
print(f"{'':<8}{'raw dur':>9}{'clean dur':>11}{'raw rms':>9}{'clean rms':>11}")
rr = np.array([r["raw_rms"] for r in rows]); cr = np.array([r["rms"] for r in rows])
rd = np.array([r["raw_seconds"] for r in rows]); cd = np.array([r["seconds"] for r in rows])
print(f"{'median':<8}{np.median(rd):8.2f}s{np.median(cd):10.2f}s{np.median(rr):9.3f}{np.median(cr):11.3f}")
print(f"{'min':<8}{rd.min():8.2f}s{cd.min():10.2f}s{rr.min():9.3f}{cr.min():11.3f}")
print(f"{'max':<8}{rd.max():8.2f}s{cd.max():10.2f}s{rr.max():9.3f}{cr.max():11.3f}")
if weak:
    print(f"\n{len(weak)} references left under 3s of speech -- weak, e.g.:")
    for r in weak[:5]:
        print(f"   {r['file']}  {r['seconds']}s")

idx = os.path.join(os.path.dirname(os.path.abspath(a.out)), "refs_clean.json")
json.dump(rows, open(idx, "w"), indent=2, ensure_ascii=False)
print(f"\nwrote {len(rows)} refs + {idx}")
