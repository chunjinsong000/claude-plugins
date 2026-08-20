#!/usr/bin/env python3
"""Synthesize every dealer_phrases_tts.json item as OVER 10 s of audio.

Nothing is trimmed - each clip is guaranteed longer than MIN_S so the caller can just
take the first 10 s. Most lines are far shorter than that, so the text is repeated:

  1. predict repeats from a per-duration_factor linear fit measured on this checkpoint
     (see COEF), aiming past MIN_S
  2. synthesize the repeated text with the item's own infer_kwargs
  3. if the take still lands under MIN_S, raise the repeat count and retry (RETRIES)
  4. only if it still falls short, loop the audio onto itself - never silence padding

Pauses between the repeated sentences come from two places, and both are needed:
  * the sentences keep their own punctuation, so the model produces its own breath
    between them (measured: 120-240 ms)
  * interval_silence inserts a hard gap BETWEEN SEGMENTS, which only happens when the
    text is actually split - so max_text_tokens_per_segment must be small enough.
    At the default seg=120 a short repeated line is never split and interval_silence
    has no effect whatsoever.

  python synth_10s.py --shard 0 --num-shards 2                # one process per GPU
  python synth_10s.py --limit 20 --out-dir /tmp/probe         # smoke test

Resumable: an output already longer than MIN_S is skipped.
"""
import argparse, json, math, os, sys, time
import numpy as np
import soundfile as sf

# Defaults sit next to this script; point --src/--out at your data instead.
HERE = os.path.dirname(os.path.abspath(__file__))
V = "/home/ubuntu/chunjin/project/valka-ai"
SRC = os.path.join(HERE, "dealer_phrases_tts.json")
OUT_DIR = os.path.join(os.path.dirname(HERE), "dealer_phrases_audio")

# dur = a*total_words + b, fitted per duration_factor on this checkpoint
# (mean |err| 0.19-0.87 s; the exact-length pass below absorbs the rest)
COEF = {0.72: (0.2147, 1.2537), 0.85: (0.3041, 1.2018), 1.0: (0.3232, 2.3319),
        1.15: (0.3781, 2.3965), 1.3: (0.3747, 2.9608)}
MIN_S = 10.4           # every clip must exceed this; caller takes the first 10 s
OVERSHOOT_S = 12.0     # what the repeat prediction aims for
MAX_REPEATS = 16
RETRIES = 3            # extra repeats added per retry when a take lands short
GAP_MS = 150           # only used by the loop fallback
FADE_MS = 25
# Pause control between repeated sentences. interval_silence is ignored unless the text
# is split, so the segment cap has to be small enough to split it.
SEG_TOKENS = 25
PAUSE_MS = 400


def predict_repeats(text, dur_f, target=OVERSHOOT_S):
    a, b = COEF.get(dur_f, COEF[1.0])
    words = max(1, len(text.split()))
    need = max(words, (target - b) / a)
    return max(1, min(MAX_REPEATS, math.ceil(need / words)))


def loop_to_min(wav, sr):
    """Last-resort padding: loop the speech (never silence) until it passes MIN_S."""
    n_min = int(math.ceil(MIN_S * sr))
    if len(wav) >= n_min:
        return wav
    fade = int(FADE_MS / 1000 * sr)
    gap = np.zeros(int(GAP_MS / 1000 * sr), dtype=wav.dtype)
    piece = wav.copy()
    if len(piece) > 2 * fade:      # crossfade the seam so the loop point does not click
        piece[:fade] *= np.linspace(0, 1, fade)
        piece[-fade:] *= np.linspace(1, 0, fade)
    out = wav.copy()
    while len(out) < n_min:
        out = np.concatenate([out, gap, piece])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--ref", default=f"{V}/tts_test_out/refs_clean/ref02.wav",
                    help="speaker reference; IndexTTS-2.5 needs no transcript for it")
    ap.add_argument("--ref-dir", default=None,
                    help="directory of reference audio; cycled across items so the corpus "
                         "carries several speakers. Overrides --ref.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--seg-tokens", type=int, default=SEG_TOKENS,
                    help="max_text_tokens_per_segment; must be small enough that the "
                         "repeated text is actually split, or interval_silence does nothing")
    ap.add_argument("--pause-ms", type=int, default=PAUSE_MS,
                    help="inter-sentence pause inserted between segments")
    a = ap.parse_args()

    refs = [a.ref]
    if a.ref_dir:
        import glob as _glob
        refs = sorted(_glob.glob(os.path.join(a.ref_dir, "*.wav")))
        if not refs:
            sys.exit(f"no .wav in --ref-dir {a.ref_dir}")

    items = json.load(open(a.src))["items"]
    items = items[a.shard::a.num_shards]
    if a.limit:
        items = items[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    manifest_path = os.path.join(a.out_dir, f"manifest_shard{a.shard}.jsonl")

    from indextts.infer_v2_5 import IndexTTS2
    tts = IndexTTS2(cfg_path=f"{V}/IndexTTS-2.5/checkpoints/config.yaml",
                    model_dir=f"{V}/IndexTTS-2.5/checkpoints", use_bf16=True)

    n_retry = n_ok = 0
    done = 0
    if os.path.exists(manifest_path):
        done = sum(1 for _ in open(manifest_path))
    mf = open(manifest_path, "a")
    tmp_path = os.path.join(a.out_dir, f".tmp_shard{a.shard}.wav")
    t_start = time.time()
    n_loop = 0

    for k, it in enumerate(items):
        out_wav = os.path.join(a.out_dir, f"{it['id']}.wav")
        if os.path.exists(out_wav):
            try:
                info = sf.info(out_wav)
                if info.frames / info.samplerate >= MIN_S:
                    continue
            except Exception:
                pass

        kw = dict(it["infer_kwargs"])
        kw.setdefault("max_text_tokens_per_segment", a.seg_tokens)
        kw["interval_silence"] = a.pause_ms      # override the mapped value: this run
                                                 # wants a deliberate inter-sentence pause
        n = predict_repeats(it["text"], kw.get("duration_factor", 1.0))
        tmp = tmp_path

        wav = sr = None
        raw = 0.0
        ref = refs[k % len(refs)]
        for attempt in range(RETRIES + 1):
            text = " ".join([it["text"]] * n)
            tts.infer(spk_audio_prompt=ref, text=text, lang=a.lang,
                      output_path=tmp, **kw)
            w, sr = sf.read(tmp)
            if w.ndim > 1:
                w = w.mean(1)
            wav = w.astype(np.float32)
            raw = len(wav) / sr
            if raw >= MIN_S or n >= MAX_REPEATS:
                break
            # short take: scale the repeat count by how far off it was, +1 minimum
            n = min(MAX_REPEATS, max(n + 1, math.ceil(n * MIN_S / max(raw, 0.1))))
            n_retry += 1

        looped = False
        if raw < MIN_S:
            wav = loop_to_min(wav, sr)
            looped = True
            n_loop += 1
        else:
            n_ok += 1
        sf.write(out_wav, wav, sr, subtype="PCM_16")

        mf.write(json.dumps({
            "id": it["id"], "phase": it["phase"], "situation": it["situation"],
            "text": it["text"], "repeats": n, "spoken_text": text,
            "raw_seconds": round(raw, 3), "seconds": round(len(wav) / sr, 3),
            "looped": looped,
            "sample_rate": sr, "path": os.path.relpath(out_wav, a.out_dir),
            "reference": ref,
            "infer_kwargs": it["infer_kwargs"],
        }) + "\n")
        mf.flush()

        if (k + 1) % 25 == 0:
            el = time.time() - t_start
            rate = (k + 1) / el
            left = (len(items) - k - 1) / rate / 3600
            print(f"[shard {a.shard}] {k+1}/{len(items)}  {rate*3600:.0f}/h  "
                  f"eta {left:.1f}h  ok={n_ok} retried={n_retry} looped={n_loop}", flush=True)

    mf.close()
    if os.path.exists(tmp_path):        # do not leave scratch files in the delivery dir
        os.remove(tmp_path)
    print(f"[shard {a.shard}] done: {len(items)} items, ok={n_ok}, retried={n_retry}, "
          f"looped={n_loop}", flush=True)


if __name__ == "__main__":
    main()
