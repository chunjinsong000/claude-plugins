---
name: corpus
description: Turn a structured phrase-bank JSON (templates with {placeholders} plus descriptive attributes like style/emphasis/length) into a fixed-length TTS corpus with IndexTTS-2.5 — expand placeholders over their value domains, convert the descriptive metadata into real control signals (emo_vector / duration_factor), concatenate different lines into clips of a guaranteed minimum length, and pair each text with several reference voices. Use when the user says "expand the placeholders", "generate the dealer phrases corpus", "make TTS training data from this JSON", "I need N clips", "every clip must be 10 seconds", "the emotion is too strong", "enthusiastic is too fast", "there is noise in this clip", "add a pause between sentences", or works with dealer_phrases.json.
---

# Phrase bank → fixed-length TTS corpus

Four stages. Built for
`DiffSynth/data/project21_snapshot_12032025_packed/dealer_phrases_text/dealer_phrases.json`
(2741 templates, 34 situations, 7 placeholders) but the shape is generic.

## The settled recipe

```bash
python fill_placeholders.py --target 120000            # 1   placeholder expansion
python make_tts_prompts.py                             # 2   attributes -> control signals
python make_mixed_prompts.py --target-clips 2000       # 2.5 concat different lines
python synth_10s.py --src dealer_phrases_mixed.json \
    --ref-dir refs_norm/ --refs-per-item 5 \
    --emo-scale 0.6 --shard $S --num-shards 2          # 3   synthesis
```

2000 texts x 5 reference voices = 10,000 clips, each >10 s. Settled after listening:
`--emo-scale 0.6`, no pause post-processing, pace steps `0.85 / 1.0 / 1.15 / 1.3`,
`emphasis: enthusiastic` gets no pace bonus. ~7.5 h on 2xH100 at 4 processes per GPU
(measured 1.85x speedup at 4-way -- the GPU saturates, so more processes buy little).

---

## Stage 1 - expand placeholders (`fill_placeholders.py`)

**Domains come from the subject's rules, not a flat numeric range.** A uniform
"points 13-30" produces `"18. That's a bust."`

| situation | domain | why |
|---|---|---|
| `*_busts` | 22-30 | must exceed 21 |
| `dealer_stands` | 17-21 | dealer stands on 17+ |
| `player_stands` | 12-21 | below 12 a player always draws |
| pair lines | 2 x rank value | **a pair of Aces is soft 12, not 22** |
| otherwise | 4-21 | hand still live |

Also: check what *follows* a placeholder (all 76 `{rank}` uses are `{rank}s`, so naive
substitution writes "Sixs" not "Sixes"); repeated placeholders in one phrase must share a
value; `"{seconds} seconds"` needs a singular guard.

`--target N` gives an exact count, water-filled evenly across templates, with global
cursors so every value still appears corpus-wide. Verify coverage, don't assume it.

---

## Stage 2 - attributes to control signals (`make_tts_prompts.py`)

Only `text` is spoken; `style`, the axes and the situation description become knobs.
`axes.addressing` maps to **nothing** - it is a property of the wording.

### Measured on the checkpoint, not assumed

- **`infer()` does not normalise `emo_vector`.** `normalize_emo_vec` (the model's
  `emo_bias` plus a hard **sum <= 0.8** cap) is only called from `webui.py`.
- **`sum(emo_vector)` is the share of the reference's own emotion being overwritten** --
  `emovec = emovec_mat + (1 - sum(weight)) * emovec_ref`. That is why the 0.8 ceiling exists.
- **`emo_vector` really works, and is monotonic**: happy 0.2/0.6/0.8 -> f0 219/245/264 Hz;
  melancholic 0.6 -> 158 Hz and quietest; surprised 0.6 -> highest f0 variance.
- **Emotion changes pace too**: happy 0.6 -> -22% duration, calm 0.6 -> -37%, while
  melancholic -> +14%. Emotion and `duration_factor` interact.
- **`duration_factor` only works in coarse steps.** `0.88-1.05` is flat *and*
  non-monotonic. Slowing is strong (+30% at 1.14), speeding up weak (-7% at 0.8).
- **Damp the wrong emotion, don't just add the right one.** Phrase banks mark losing
  lines `emphasis: enthusiastic` meaning *forceful*; adding melancholic still left
  `happy 0.47` on `"Bust at 22!"`. Sympathy situations multiply happy by 0.25 first.
- **Pace steps are `0.85 / 1.0 / 1.15 / 1.3`** -- `0.72` was removed as rushed, and
  `enthusiastic` no longer buys a step. Together that moved those lines two steps slower.
  Dropping the fastest step only affects `score = -3` rows; every other score is untouched
  (calm's distribution was byte-identical before and after).

Emotion intensity is the knob users actually ask for. `--emo-scale` (via `emo_alpha`, no
regeneration needed): 1.0 -> f0 235 Hz, **0.6 -> 198 Hz (settled)**, 0.4 -> 180, off -> 171.

---

## Stage 2.5 - concatenate different lines (`make_mixed_prompts.py`)

Repeating one line for 10 s is mechanical. Two non-obvious requirements:

- **One line per `text_template` per clip.** Stage 1 expands each template into dozens of
  placeholder siblings, so a naive concat gives *"Unfortunately, 27 is a bust.
  Unfortunately, 28 is a bust. ..."* -- same wording, different number. Deduplicating by
  template yields real variation instead.
- **Group by `phase+situation+style+emphasis`**, so one control signal fits every line in
  the clip. Residual spread inside a group comes only from `sentence_type`; the emitted
  vector is the group median. Strict equality would need 524 groups, only 379 long enough.

Concatenating also measured *cleaner* than repeating (flatness 0.003-0.022 vs 0.016-0.020),
partly because distinct lines reach 10 s with fewer tokens than a repeated short line.

### Quotas must go by TEMPLATE count, not phrase count

Placeholder combinatorics are wildly uneven -- **99.1% of the cartesian capacity sits in
neutral situations** (a card+box+points line has 6,552 combinations; `"Winner!"` has 1).
Quotas by phrase count gave **96% neutral clips against 47% of the templates**. Template
count reflects how the bank was actually written, so that is the mix to reproduce.

The consequence is a hard three-way tradeoff -- non-neutral capacity is tiny
(celebrate 1,193 vs neutral 480,077), so *10,000 clips + correct emotion mix + no text
reuse* cannot hold at once:

| want | get |
|---|---|
| correct mix, zero reuse | ~400 clips |
| correct mix, 2.0x reuse | **2,000 clips (settled)** |
| correct mix, 8.6x reuse | 10,000 clips |
| zero reuse, any mix | 9,349 clips but 96% neutral |

**When audio variety comes from many reference voices, keep the text set small.** With
10,000 references, 2000 texts x `--refs-per-item 5` uses every reference exactly once and
holds phrase reuse at 2.0x, versus 8.6x for a 10,000-text set. Reuse grows slowly from
400->4000 clips (1.4x->2.5x) because neutral capacity is ample; it only explodes at 10,000.

---

## Stage 3 - synthesis (`synth_10s.py`)

Clips are guaranteed **over** `MIN_S` and never trimmed, so the caller takes the first 10 s.
Repeats are predicted from a per-`duration_factor` linear fit (`COEF`), retried if short,
and waveform looping exists only as a last resort (never triggered in practice).

### Segmentation destroys the audio - this is the big one

`interval_silence` only inserts a gap **between segments**, and segmentation only happens
above `max_text_tokens_per_segment`. Forcing a split to get pauses (seg=25 -> 3 segments)
wrecked the output: **spectral flatness 0.106 vs 0.014** for the same text as one segment,
with stretches of pure noise (flatness p90 hit 1.0). An apparent "too many repeats"
threshold turned out to be exactly the 1-segment -> 2-segment boundary (6-word line: rep3
= 21 tokens clean, rep4 crosses 25 and breaks; 2-word line flips at rep5->rep6).

**Keep `--seg-tokens` at 120 and the text under ~70 words.** Pauses are not worth
segmenting for: stage 2.5's concatenation already leaves natural 100-360 ms gaps, and
stretching them to a uniform 600 ms in post measured slightly *worse*
(flatness 0.0197 vs 0.0161) with no gain in gap count. `--pause-ms 0` is the default. The
stretcher remains for cases needing deliberate long beats -- note it finds nothing to
stretch on the slow pace step, where the model runs sentences together with breath filling
the joins (4/10 clips had no gap above the -45 dB floor, and -40 dB did not help).

### Reference loudness transfers to the output

Quiet reference in, quiet clone out: three avspeech refs at rms 0.014-0.022 produced clips
at 0.015-0.070. Normalizing (`prep_refs.py` in the sibling `synthesize` skill) lifted the
same clips to 0.124-0.225. **Always normalize references first** -- and with thousands of
them, measure the distribution before deciding whether to normalize all or just outliers.

`--refs-per-item K` renders each text with K different voices, assigning
`refs[i*K + j]`, so `len(refs) == n_texts * K` consumes each reference exactly once.

---

## Verification

- **ASR the first 10 s** (the slice actually consumed), and expect repeats -- check the
  line appears at least once rather than diffing the whole string. Whisper must be pinned
  to `temperature=0.0, condition_on_previous_text=False`; the default fallback invents
  failures (same file scored 100% then 58.8%; an 8 s clip read as 25 repetitions).
- **Spectral flatness is the noise detector.** Reference audio sits near 0.0197; a good
  clip matches it, a segmented one hits 0.08-0.12.
- **rms per clip** catches the quiet-reference problem, invisible in a file listing.
- **Single-clip pace comparisons are worthless.** Generation is re-sampled every run, so
  words/sec moved +-40% between runs with *identical* parameters -- one clip went
  3.55 -> 1.94 wps with nothing changed. Judge pace from the `duration_factor`
  distribution over the whole set, and settle "is it too fast" by ear.
- **Don't vary emotion across repeats.** Three passes with a decaying `--emo-scale` cost
  3x and delivered nothing recoverable: f0 came out 285->198->255->209->232->195 Hz because
  per-repeat variation (+-35 Hz) swamps the ~20 Hz the scale ladder contributes. One pass
  already spreads f0 64.7 Hz vs 90.1 Hz for three.
- **The manifest must record the kwargs actually used**, not the mapped ones -- otherwise
  `--emo-scale` and the pace compensation are invisible and the run is unreproducible.
  `synth_10s.py` writes both (`infer_kwargs` and `infer_kwargs_mapped`).
