---
name: corpus
description: Turn a structured phrase-bank JSON (templates with {placeholders} plus descriptive attributes like style/emphasis/length) into a fixed-length TTS corpus with IndexTTS-2.5 — expand placeholders over their value domains, convert the descriptive metadata into real control signals (emo_vector / duration_factor), concatenate different lines into clips of a guaranteed minimum length, and pair each text with several reference voices. Use when the user says "expand the placeholders", "generate the dealer phrases corpus", "make TTS training data from this JSON", "I need N clips", "every clip must be 10 seconds", "the emotion is too strong", "enthusiastic is too fast", "there is noise in this clip", "add a pause between sentences", or works with dealer_phrases.json.
---

# Phrase bank → fixed-length TTS corpus

Five stages. Built for
`DiffSynth/data/project21_snapshot_12032025_packed/dealer_phrases_text/dealer_phrases.json`
(2741 templates, 34 situations, 7 placeholders) but the shape is generic.

## The settled recipe (v2: unique texts, LLM-augmented bank)

```bash
python paraphrase_templates.py --variants 14 --batch 16 \
    --out dealer_phrases_aug.json                        # 0.5 grow the bank with an LLM
python fill_placeholders.py --src dealer_phrases_aug.json \
    --target 120000 --out dealer_phrases_filled.json     # 1   placeholder expansion
python make_tts_prompts.py                               # 2   attributes -> control signals
python make_mixed_prompts.py --target-clips 10000        # 2.5 concat different lines
python synth_10s.py --src dealer_phrases_mixed.json \
    --ref-dir refs_norm/ --refs-per-item 1 \
    --emo-scale 0.6 --shard $S --num-shards 8            # 3   synthesis
```

10,000 unique texts x 1 reference voice each = 10,000 clips, every clip >10 s, every one
of the 2,737 native templates present, emotion mix exactly the original bank's. Settled
after listening: `--emo-scale 0.6`, no pause post-processing, pace steps
`0.85 / 1.0 / 1.15 / 1.3`, `emphasis: enthusiastic` gets no pace bonus. ~7.5 h on 2xH100
at 4 processes per GPU (1.85x speedup at 4-way; the GPU saturates beyond that).

Earlier v1 variant (no LLM augmentation): 2,000 texts x `--refs-per-item 5` — same audio
count, texts repeat across 5 voices, phrase reuse 2.0x. Use it when no LLM is available.

---

## Stage 0.5 - grow the bank with paraphrases (`paraphrase_templates.py`)

Non-neutral capacity is the binding constraint (celebrate: 561 templates, 1,193
combinations), so real text diversity needs new WORDINGS, not more placeholder values.
A local instruction-tuned LLM works fine — this used gemma-3-12b-it (LTX's text encoder,
already on disk), batch 16, ~50 min for 2,741 templates x 14 variants on one H100.

The generation is the easy half. The filters are where the correctness lives, and each
one exists because the unfiltered output actually contained the failure:

1. **Invented placeholders.** The model produced `{player name}` (space!) and
   `{round number}` — nothing in the bank. The obvious check, comparing
   `re.findall(r"\{[a-z_]+\}")` multisets, PASSES these, because the malformed forms
   don't match the pattern on either side. Check instead that every `\{[^}]*\}` span is
   in the exact whitelist of valid placeholders, and that braces balance. 530 dropped —
   and these would have been spoken aloud as "curly brace player name".
2. **Bare-context placeholders.** `{box}` without a seat word before it reads as "for 3";
   `{seconds}` without "seconds" after it reads as "closing in 3". 137 dropped.
3. **Bookish register.** "Twenty-one, surpassed." passes every mechanical check. A second
   LLM pass judging "natural SPOKEN dealer English? YES/NO" (greedy, 3 tokens, batch 96,
   ~10 min) dropped 5,239 of 27,340 (19%).

Dedup must be **situation-wide and single-definition**: per-template dedup let two
neighbouring templates both produce "Bets are open, please.", and one variant regenerated
another template's original verbatim; then a second, subtly different key function
(missing `.strip()`) still let 4 through. One `norm_key()`, pre-seeded with every original
line in the situation.

Originals are always kept; variants carry `origin: "paraphrase"` + `origin_index`.
Yield after all filters: ~8 kept of 14 requested per template (2,741 -> 24,842).

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

### Quotas must go by NATIVE template count, not phrase count

Placeholder combinatorics are wildly uneven -- **99.1% of the cartesian capacity sits in
neutral situations** (a card+box+points line has 6,552 combinations; `"Winner!"` has 1).
Quotas by phrase count gave **96% neutral clips against 47% of the templates**. And after
LLM augmentation, quota by *all*-template count still drifts a couple of points, because
the paraphrase yield is uneven after filtering — quota by **native** template count
reproduces the original structure exactly (measured 47.1/20.4/14.7/11.9/5.9 vs
47.1/20.5/14.7/11.9/5.8).

Without stage 0.5 there is a hard three-way tradeoff (non-neutral capacity is tiny:
celebrate 1,193 vs neutral 480,077) — *10,000 clips + correct mix + no text reuse* cannot
hold, and the best compromise is 2,000 texts at 2.0x reuse, multiplied out by
`--refs-per-item 5`. Stage 0.5 dissolves the tradeoff: 10,000 unique texts at the correct
mix, phrase reuse 1.6x.

Two more requirements that do not fall out automatically:

- **"Include every native template" needs an explicit walk order.** Proportional quotas
  alone covered only 33.6% of the native templates. Fix: per group, keep a
  `native_pending` set and put those templates first in each clip's walk; then shuffle the
  chosen lines (natives-first would otherwise front-load them, and the previous sorted
  walk clustered same-prefix paraphrases: "A natural! A natural! A natural...").
- **Values differ WITHIN a clip** (a dealer_stands clip can say "20... 21... 19"), because
  each line's placeholders were filled independently at stage 1. Fine for voice/prosody
  training — flag it to the user rather than silently accepting it.

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
**The index must be assigned before sharding**: the fallback (the loop counter) is a
per-shard index, so with 8 shards every shard reused refs[0:1250] and 8,750 references
were never touched. Verify from the manifests: distinct-reference count == reference-file
count, min uses == max uses == K/n ratio.

### Long background jobs: `nohup` is NOT enough

Twice in this project a multi-hour run silently died mid-flight with no traceback: the
harness's Bash-tool timeout kills the whole process group, and `nohup` only blocks
SIGHUP. The signature is workers all dying at the same wall-clock moment with logs that
just stop. Launch as
`setsid env CUDA_VISIBLE_DEVICES=$G nohup python ... < /dev/null >> log 2>&1 &`
so each worker is its own session — verified to survive even the parent CLI process
exiting. Keep the launcher call itself fast; put waiting in a separate monitor command.

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
