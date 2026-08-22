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
# Pauses are NOT made with interval_silence. Measured: interval_silence only applies
# between segments, and forcing a split (seg=25 -> 3 segments) wrecks the audio --
# spectral flatness 0.106 vs 0.014 for the same text generated as one segment, with
# stretches of pure noise (flatness p90 hit 1.0). The apparent "too many repeats"
# threshold was really the 1-segment -> 2-segment boundary. So: synthesize unsegmented,
# then stretch the model's own inter-sentence gaps in post. Fully controllable, no artifacts.
SEG_TOKENS = 120       # the model default: keep the text in ONE segment
PAUSE_MS = 0           # settled default: no pause post-processing at all. Concatenating
                       # different lines (stage 2.5) already leaves natural 100-360 ms
                       # gaps, and stretching them to a uniform 600 ms measured slightly
                       # WORSE (flatness 0.0197 vs 0.0161) for no gain in gap count.
# Gap detection floor. At -45 dB the fast pace steps (0.72/0.85) show clear 100-600 ms
# inter-sentence gaps, but the slow step (1.15) runs the sentences together with
# low-level breath filling the joins, so nothing is found to stretch. -40 dB finds gaps
# in most of those too; -35 dB starts catching intra-word gaps (14 per clip), so do not
# go below -40 without listening.
PAUSE_FLOOR_DB = -45.0

# Emotion speeds the delivery up, but only on some dims. Measured against a no-emotion
# baseline of 4.27 s on one line: happy 0.6 -> 3.34 s (-22%), calm 0.6 -> 2.67 s (-37%),
# while surprised 0.6 -> 4.39 s (+3%) and melancholic 0.6 -> 4.86 s (+14%).
# Compensation is applied by stepping the pace ladder, since duration_factor is flat
# within +-0.05 and a nudge would do nothing.
# NOTE: calm measures as the *fastest* dim, but compensating it made the calm lines drag --
# they already sit high on the ladder (1.15) and a step took them to 1.30. So only happy
# is compensated; the calm lines keep their mapped pace.
# And the happy threshold alone also caught urgent/dealing lines whose emphasis is
# "enthusiastic" -- those are *meant* to be quick. So the class gate is required too:
# stage 2 tags each item with source.emotion_class, which is the single definition.
ACCEL_DIMS = ("happy",)
ACCEL_THRESHOLD = 0.30
ACCEL_CLASSES = ("celebrate",)
PACE_LADDER = [0.85, 1.00, 1.15, 1.30]   # 0.72 dropped: too rushed
EMO_DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def predict_repeats(text, dur_f, target=OVERSHOOT_S):
    a, b = COEF.get(dur_f, COEF[1.0])
    words = max(1, len(text.split()))
    need = max(words, (target - b) / a)
    return max(1, min(MAX_REPEATS, math.ceil(need / words)))


def stretch_pauses(wav, sr, target_ms=PAUSE_MS, min_ms=90, thr_db=PAUSE_FLOOR_DB):
    """Lengthen the gaps the model already leaves between repeated sentences.

    Done in post because interval_silence needs segmentation, and segmentation degrades
    the audio (see the note above). Only gaps already >= min_ms are stretched, so pauses
    inside a word are left alone. Leading/trailing silence is untouched.

    min_ms is 90, not 150: unsegmented output leaves inter-sentence gaps as short as
    ~100 ms, and a 150 ms floor silently skipped half the clips. Intra-word gaps are
    well under 90 ms, so they are still safe.
    """
    if target_ms <= 0:
        return wav
    fr = max(1, int(0.02 * sr))
    n = len(wav) // fr
    if n < 3:
        return wav
    frames = wav[: n * fr].reshape(n, fr)
    quiet = np.sqrt((frames ** 2).mean(1)) < 10 ** (thr_db / 20)

    runs = []                       # (start_frame, end_frame) of internal quiet runs
    i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if i > 0 and j < n and (j - i) * 20 >= min_ms:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return wav

    target_frames = int(round(target_ms / 20))
    out, prev = [], 0
    for a, b in runs:
        out.append(wav[prev * fr: b * fr])
        extra = target_frames - (b - a)
        if extra > 0:                       # pad with the quietest part of the gap itself,
            gap = wav[a * fr: b * fr]       # so room tone continues instead of hard silence
            tile = np.tile(gap, int(np.ceil(extra / max(len(gap) // fr, 1))))[: extra * fr]
            out.append(tile)
        prev = b
    out.append(wav[prev * fr:])
    return np.concatenate(out).astype(wav.dtype)


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
                    help="directory of reference audio. Overrides --ref.")
    ap.add_argument("--refs-per-item", type=int, default=1,
                    help="how many different reference voices to render each text with. "
                         "Output ids get a _r<NN> suffix. With N texts x K refs-per-item "
                         "and exactly N*K references available, every reference is used "
                         "once -- which is the point: variety comes from the voices, so the "
                         "text set can stay small and near-unique.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lang", default="EN")
    ap.add_argument("--seg-tokens", type=int, default=SEG_TOKENS,
                    help="max_text_tokens_per_segment. Keep it HIGH (default 120) so the "
                         "text stays in one segment; splitting degrades the audio badly.")
    ap.add_argument("--pause-ms", type=int, default=PAUSE_MS,
                    help="target inter-sentence gap, stretched in post-processing "
                         "(0 disables). NOT interval_silence -- see the note in this file.")
    ap.add_argument("--no-emo-pace-compensate", dest="emo_pace_compensate",
                    action="store_false",
                    help="disable the one-step pace slowdown applied when happy+calm "
                         "weight is high; emotion otherwise speeds the delivery up")
    ap.add_argument("--pause-floor-db", type=float, default=PAUSE_FLOOR_DB,
                    help="quiet threshold for finding gaps to stretch. -45 (default) only "
                         "finds them on the fast pace steps; -40 also reaches most slow-step "
                         "lines; below -40 it starts stretching intra-word gaps.")
    ap.add_argument("--emo-scale", type=float, default=1.0,
                    help="global emotion intensity, 0-1. Scales emo_vector via emo_alpha, "
                         "so it needs no regeneration of the prompt json. sum(emo_vector) is "
                         "the share of the reference's own emotion that gets overwritten, so "
                         "0.5 here halves how far each line departs from the reference voice.")
    a = ap.parse_args()

    refs = [a.ref]
    if a.ref_dir:
        import glob as _glob
        refs = sorted(_glob.glob(os.path.join(a.ref_dir, "*.wav")))
        if not refs:
            sys.exit(f"no .wav in --ref-dir {a.ref_dir}")

    items = json.load(open(a.src))["items"]
    # _ref_index must be assigned BEFORE sharding, for every refs_per_item value: the
    # fallback (the loop counter k) is a per-shard index, so with 8 shards every shard
    # would reuse refs[0:len(shard)] and most references would never be touched.
    for i, it in enumerate(items):
        it["_ref_index"] = i * a.refs_per_item
    if a.refs_per_item > 1:
        # expand each text into one job per reference voice, assigning references so that
        # a run with len(refs) == len(items)*refs_per_item consumes each reference exactly once
        expanded = []
        for i, it in enumerate(items):
            for j in range(a.refs_per_item):
                job = dict(it)
                job["id"] = f"{it['id']}_r{j:02d}"
                job["_ref_index"] = i * a.refs_per_item + j
                expanded.append(job)
        items = expanded
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
        kw["interval_silence"] = 0               # pauses are added in post, not here
        if "emo_vector" in kw and a.emo_scale != 1.0:
            # emo_alpha scales the vector inside infer(); clamped to [0,1] there
            kw["emo_alpha"] = max(0.0, min(1.0, a.emo_scale))
        emo_class = (it.get("source") or {}).get("emotion_class", "neutral")
        if a.emo_pace_compensate and "emo_vector" in kw and emo_class in ACCEL_CLASSES:
            ev = kw["emo_vector"]
            accel = sum(ev[EMO_DIMS.index(d)] for d in ACCEL_DIMS) * kw.get("emo_alpha", 1.0)
            if accel > ACCEL_THRESHOLD:
                cur = kw.get("duration_factor", 1.0)
                i = min(range(len(PACE_LADDER)), key=lambda j: abs(PACE_LADDER[j] - cur))
                kw["duration_factor"] = PACE_LADDER[min(i + 1, len(PACE_LADDER) - 1)]
        n = predict_repeats(it["text"], kw.get("duration_factor", 1.0))
        tmp = tmp_path

        wav = sr = None
        raw = 0.0
        ref = refs[it.get("_ref_index", k) % len(refs)]
        for attempt in range(RETRIES + 1):
            text = " ".join([it["text"]] * n)
            tts.infer(spk_audio_prompt=ref, text=text, lang=a.lang,
                      output_path=tmp, **kw)
            w, sr = sf.read(tmp)
            if w.ndim > 1:
                w = w.mean(1)
            wav = w.astype(np.float32)
            raw = len(wav) / sr
            stretched_wav = stretch_pauses(wav, sr, target_ms=a.pause_ms,
                                           thr_db=a.pause_floor_db)
            stretched = len(stretched_wav) / sr
            if stretched >= MIN_S or n >= MAX_REPEATS:
                break
            # short take: scale the repeat count by how far off it was, +1 minimum
            n = min(MAX_REPEATS, max(n + 1, math.ceil(n * MIN_S / max(stretched, 0.1))))
            n_retry += 1

        wav = stretched_wav
        looped = False
        if stretched < MIN_S:
            wav = loop_to_min(wav, sr)
            looped = True
            n_loop += 1
        else:
            n_ok += 1
        sf.write(out_wav, wav, sr, subtype="PCM_16")

        mf.write(json.dumps({
            "id": it["id"], "phase": it["phase"], "situation": it["situation"],
            "text": it["text"], "repeats": n, "spoken_text": text,
            "raw_seconds": round(raw, 3), "stretched_seconds": round(stretched, 3),
            "seconds": round(len(wav) / sr, 3),
            "looped": looped, "emotion_class": emo_class,
            "sample_rate": sr, "path": os.path.relpath(out_wav, a.out_dir),
            "reference": ref,
            "infer_kwargs": kw,             # what was ACTUALLY used, incl. --emo-scale
                                            # and the pace compensation, so it reproduces
            "infer_kwargs_mapped": it["infer_kwargs"],
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
