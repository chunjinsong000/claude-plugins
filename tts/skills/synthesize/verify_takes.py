#!/usr/bin/env python3
"""Round-trip every generated wav through Whisper and report what actually came out.

Catches the failure modes you cannot hear from a file listing: near-silent output,
dropped/repeated sentences, and runaway length. Run this on every batch --
during this project's five-engine comparison, every "failure" flagged under default
Whisper decoding turned out to be an ASR artifact, and two real failures (a silent
VoxCPM2 take, a 655s Qwen take) were only visible here.

  python verify_takes.py --dir out/ --text-file line.txt

IMPORTANT: uses temperature=0.0, condition_on_previous_text=False. Whisper's default
temperature-fallback is non-deterministic -- the same file scored 100% on one pass and
58.8% on another, and once transcribed an 8s clip as 25 repetitions of the sentence.
"""
import argparse, difflib, glob, json, os
import numpy as np, soundfile as sf
from faster_whisper import WhisperModel

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="directory of wavs to score")
ap.add_argument("--text", help="the line the engine was asked to speak")
ap.add_argument("--text-file", help="file holding that line")
ap.add_argument("--model", default="large-v3")
ap.add_argument("--lang", default="en")
ap.add_argument("--json-out", default=None)
ap.add_argument("--expect-seconds", type=float, default=None,
                help="flag takes longer than this (runaway detection)")
a = ap.parse_args()
ref = a.text or open(a.text_file).read().strip()
norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").split()
NW = len(norm(ref))

m = WhisperModel(a.model, device="cuda", compute_type="float16")
files = sorted(glob.glob(os.path.join(a.dir, "*.wav")))
out = {}
print(f"reference: {NW} words\n")
print(f"{'file':<40}{'dur':>7}{'wps':>6}{'rms':>7}{'sim':>7}  flags")
print("-" * 84)
for f in files:
    w, sr = sf.read(f)
    w = w if w.ndim == 1 else w.mean(1)
    if w.dtype not in (np.float32, np.float64):
        w = w.astype(np.float32) / 32768
    dur = len(w) / sr
    rms = float(np.sqrt((w ** 2).mean()))
    segs, _ = m.transcribe(f, beam_size=5, language=a.lang,
                           temperature=0.0, condition_on_previous_text=False)
    t = " ".join(s.text.strip() for s in segs)
    sim = difflib.SequenceMatcher(None, norm(ref), norm(t)).ratio()
    flags = []
    if rms < 0.02: flags.append("TOO QUIET")
    if sim < 0.95: flags.append("CONTENT")
    if a.expect_seconds and dur > a.expect_seconds: flags.append("RUNAWAY")
    out[os.path.basename(f)[:-4]] = {"dur": dur, "rms": rms, "sim": sim, "sr": int(sr), "text": t}
    print(f"{os.path.basename(f):<40}{dur:6.2f}s{NW/dur:6.2f}{rms:7.3f}{sim*100:6.1f}%  "
          f"{','.join(flags)}")
    if flags:
        print(f"    heard: {t[:150]}")
if a.json_out:
    json.dump(out, open(a.json_out, "w"), indent=2)
    print(f"\nwrote {a.json_out}")
bad = sum(1 for v in out.values() if v["sim"] < 0.95 or v["rms"] < 0.02)
print(f"\n{len(out)} takes, {bad} flagged")
