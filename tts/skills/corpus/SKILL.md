---
name: corpus
description: Turn a structured phrase-bank JSON (templates with {placeholders} plus descriptive attributes like style/emphasis/length) into a fixed-length TTS corpus with IndexTTS-2.5 — expand placeholders over their value domains, convert the descriptive metadata into real control signals (emo_vector / duration_factor / interval_silence), and synthesize clips of a guaranteed minimum length by repeating the line. Use when the user says "expand the placeholders", "generate the dealer phrases corpus", "make TTS training data from this JSON", "I need N lines / N clips", "every clip must be 10 seconds", "add a pause between the repeats", "turn these attributes into TTS control signals", or works with dealer_phrases.json.
---

# Phrase bank → fixed-length TTS corpus

Three stages, each a script here. Built for
`DiffSynth/data/project21_snapshot_12032025_packed/dealer_phrases_text/dealer_phrases.json`
(2741 templates, 34 situations, 7 placeholders) but the shape is generic: templates with
`{placeholders}` plus descriptive axes.

```bash
python fill_placeholders.py --src phrases.json --out filled.json --target 10000
python make_tts_prompts.py  --src filled.json  --out tts.json
python synth_10s.py --src tts.json --ref-dir refs_norm/ --out-dir audio/ --shard 0 --num-shards 2
```

Defaults point next to each script; pass `--src/--out` for your own data. Stage 3 is
resumable and shards across GPUs.

---

## Stage 1 — expand placeholders (`fill_placeholders.py`)

**Domains must come from the subject's rules, not from a flat numeric range.** A range
like "points 13-30" applied uniformly produces `"18. That's a bust."` — semantically
broken. Look at which situation each placeholder appears in first:

| situation | domain | why |
|---|---|---|
| `*_busts` | 22–30 | "that's a bust" must exceed 21 |
| `dealer_stands` | 17–21 | dealer stands on 17+ |
| `player_stands` | 12–21 | below 12 a player always draws |
| pair lines (`{rank}`+`{points}`) | 2 × rank value | and **a pair of Aces is soft 12, not 22** — 22 is a bust, contradicting the split line it sits in |
| otherwise | 4–21 | hand still live |

Other traps found by inspecting the templates before writing any code:

- **Check what follows each placeholder.** All 76 `{rank}` occurrences are `{rank}s`
  ("a pair of {rank}s"), so a naive substitution yields "Sixs". Replace `{rank}s` with a
  real plural table (Six→**Sixes**) *before* the bare `{rank}` rule.
- **Repeated placeholders in one phrase must share a value.**
  `"Shall we call it even? {points} to {points}."` has to print the same number twice.
- **Singular/plural on numbers.** `"{seconds} seconds"` becomes "1 seconds" unless
  handled (here the domain simply avoids 1).

### Sizing the output

- **no `--target`** → every template emitted once per value; multi-placeholder phrases
  rotate **in lockstep**, not a cartesian product, so `"{card} for box {box}."` gives 52
  lines (all 52 cards, boxes cycling) not 364. On this data: 21,597 lines.
- **`--target N`** → exact count. Placeholder-free templates give 1 line each; the rest is
  **water-filled evenly** across templates up to each one's cartesian capacity, so a
  high-capacity template (6,552 for a card+box+points line) cannot flood the corpus.
- **Coverage becomes corpus-wide** under a tight target. With ~8 lines per template you
  cannot fit 52 cards per template, so placeholder values rotate on **global cursors** —
  every value still appears across the corpus. At `--target 10000`: 52/52 cards (43–44
  uses each), 13/13 ranks, 7/7 boxes, 10/10 names. Always verify this, don't assume it.

Capacity on this data: floor 2,741 (one per template), lockstep 21,597, cartesian 482,622.

---

## Stage 2 — descriptive attributes → control signals (`make_tts_prompts.py`)

Only `text` is spoken. `style`, the axes, and the situation description are *descriptions*
and must be mapped to knobs the engine actually has:

| source | becomes |
|---|---|
| `axes.emphasis` | `emo_vector` (main driver) + `duration_factor` |
| `style` | `emo_vector` (professional = calmer, friendly = warmer) |
| `axes.sentence_type` | `emo_vector` + `duration_factor` |
| `axes.directness`, `axes.length` | `duration_factor`, `interval_silence` |
| situation | `emo_vector` (wins celebrate, losses sympathise) |
| `axes.addressing` | **nothing** — it is a property of the wording, not the sound |

Output items carry `infer_kwargs` usable verbatim:
`tts.infer(spk_audio_prompt=REF, text=item["text"], lang="EN", output_path=out, **item["infer_kwargs"])`

### Three things measured on the checkpoint, not assumed

1. **`IndexTTS2.infer()` does not normalise `emo_vector`.** `normalize_emo_vec` — which
   applies the model's own `emo_bias` and a hard **sum ≤ 0.8** cap — is only called from
   `webui.py`. Apply both yourself or you are feeding the model out-of-range vectors.
2. **`duration_factor` only works in coarse steps.** Measured, one fixed text:
   `0.7→3.34s 0.8→4.14s 0.88→4.42s 0.95→4.28s 1.0→4.47s 1.05→4.64s 1.14→5.82s 1.3→6.33s`.
   The **0.88–1.05 band is flat and non-monotonic** (0.88 came out *longer* than 0.95).
   Slowing is strong (+30% at 1.14), speeding up is weak (−7% at 0.8). Score the axes as an
   integer and quantise onto audible steps: **0.72 / 0.85 / 1.00 / 1.15 / 1.30**.
3. **Damp the wrong emotion, don't just add the right one.** Phrase banks mark losing lines
   as `emphasis: enthusiastic` meaning *forceful*, not *cheerful*. Adding melancholic on top
   of an enthusiastic happy base still gave `happy 0.47` on `"Bust at 22!"` — a gleeful
   bust. Sympathy situations **multiply** happy by 0.25 before adding melancholic:
   → `happy 0.13, melancholic 0.11, surprised 0.15`, while `"Winner!"` keeps `happy 0.60`.

For a dealer voice only happy / surprised / calm (+ a little melancholic) are useful;
angry / afraid / disgusted stay 0. Dim order is fixed:
`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.

---

## Stage 3 — fixed-length synthesis (`synth_10s.py`)

Lines are 1–12 words; clips must be ≥10 s. The text is **repeated**, and the length is
guaranteed rather than trimmed (`MIN_S = 10.4`), so the caller can just take the first 10 s.

1. predict repeats from a per-`duration_factor` linear fit of duration vs word count
   (`COEF`, measured on this checkpoint — refit if the checkpoint changes)
2. synthesize; if the take lands short, raise the repeat count and retry
3. loop the waveform only as a last resort — on the 10-line probe it was never needed

### Pauses between repeats need BOTH layers

- **text layer** — keep each repeat's punctuation; the model then breathes between them
  (measured 120–240 ms).
- **inference layer** — `interval_silence` inserts a hard gap **only between segments**,
  and segmentation only happens when the text exceeds `max_text_tokens_per_segment`.
  At the default 120 a short repeated line is never split, so **`interval_silence` does
  nothing at all**. Force it: `--seg-tokens 25 --pause-ms 400` → 2–5 pauses of
  180–700 ms in the first 10 s.

### Reference loudness transfers to the output

Quiet reference in, quiet clone out. Three avspeech references at rms 0.014–0.022 produced
clips at rms 0.015–0.070. Normalizing them (desilence + rms→0.08 with the sibling
`synthesize` skill's `prep_refs.py`) lifted the same clips to 0.124–0.225 and evened out
the whole set. **Normalize references before any full run.**
`--ref-dir` cycles a directory of references so the corpus carries several speakers.

---

## Verification (do this every time)

```bash
# content, on exactly the slice the consumer will use (the first 10 s), not the whole clip
python -c "...faster_whisper, temperature=0.0, condition_on_previous_text=False..."
```

- **ASR the first 10 s**, not the full clip — that is what gets consumed. Expect the line
  to repeat; check it appears at least once rather than diffing the whole string. Allow for
  ASR rewriting "box 1" as "box one" and dropping a leading "Ah,".
- **Whisper must be pinned to `temperature=0.0, condition_on_previous_text=False`.** Under
  default temperature-fallback the same file scored 100% on one pass and 58.8% on another.
- **rms per clip** — catches the quiet-reference problem, which is inaudible in a file listing.
- **silence-run histogram** — confirms the inter-sentence pauses actually exist.
- **placeholder domain coverage** — assert every value appears; a tight `--target` silently
  drops values if the global cursors are not wired up.

## Cost

10,000 clips of 12–20 s each: roughly 15–20 h across 2 H100s. Stage 3 is resumable (a clip
already over `MIN_S` is skipped), shards with `--shard/--num-shards`, and prints an ETA
every 25 items. Confirm with the user before launching a run of that size.
