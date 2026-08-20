---
name: synthesize
description: Run any of the five TTS engines deployed under valka-ai — Qwen3-TTS, VoxCPM2, FireRedTTS3, Fish Audio S2 Pro, IndexTTS-2.5 — for zero-shot voice cloning, natural-language voice design, and pace/pitch control, plus the reference-audio prep and ASR + CAM++ verification that tells you whether a take is actually good. Use when the user says "run TTS", "synthesize this line", "clone this voice", "use IndexTTS / VoxCPM / FireRedTTS / Fish S2 / Qwen3-TTS", "compare the TTS engines", "make a male/female voice at slow/fast pace", "score the speaker similarity", "why is the cloned audio silent / too long", or asks which TTS to pick for live-dealer speech.
---

# Five TTS engines — run, control, verify

Five engines live side by side under `VALKA_ROOT` (default
`/home/ubuntu/chunjin/project/valka-ai`), each in **its own venv** — their
transformers pins conflict (4.52.1 / 4.57.3 / 5.6.2 / 5.15.1), so never share one.
`tts.sh` picks the right interpreter for you.

| engine | repo dir | model | sr | needs ref transcript? | reproducible? |
|---|---|---|---|---|---|
| `indextts25` | `IndexTTS-2.5` | IndexTeam/IndexTTS-2.5 (0.8B) | 22.05 kHz | **no** | yes |
| `voxcpm2` | `VoxCPM2` | openbmb/VoxCPM2 (2B) | 48 kHz | yes (for clone) | yes (`seed`) |
| `fireredtts3` | `FireRedTTS3` | FireRedTeam/FireRedTTS3 | 24 kHz | yes (Base) | yes (deterministic) |
| `fishs2pro` | `FishS2` | fishaudio/s2-pro (4B) | 44.1 kHz | yes | yes (`seed`) |
| `qwen3tts` | `Qwen3-TTS` | Qwen3-TTS-12Hz-1.7B ×3 | 24 kHz | yes (clone) | **no — no seed** |

## Run one line

```bash
# clone a voice (indextts25 is the only one that needs no transcript)
./tts.sh indextts25 --text-file line.txt --ref ref.wav --out out.wav

# clone with transcript (every other engine)
./tts.sh fireredtts3 --text-file line.txt --ref ref.wav --ref-text "what ref.wav says" --out out.wav
./tts.sh fishs2pro   --text-file line.txt --ref ref.wav --ref-text "..."               --out out.wav
./tts.sh voxcpm2     --text-file line.txt --ref ref.wav --ref-text "..."               --out out.wav

# voice design from a description, no reference audio
./tts.sh voxcpm2     --text "..." --instruct "A young woman, clear and confident voice" --out out.wav
./tts.sh fireredtts3 --text "..." --instruct "A middle-aged man, unhurried and even"   --out out.wav
./tts.sh qwen3tts    --text "..." --instruct "A calm male dealer, moderate pace"        --out out.wav

# built-in speaker (qwen3tts only): vivian serena uncle_fu dylan eric ryan aiden ono_anna sohee
./tts.sh qwen3tts --text "..." --speaker Ryan --out out.wav
```

`CUDA_VISIBLE_DEVICES` is honored (defaults to 0). `--max-seconds` hard-truncates —
for `qwen3tts` it is the **only** reliable length bound (see below). The engine choice
also picks the checkpoint for `qwen3tts`: `--speaker` → CustomVoice, `--instruct` →
VoiceDesign, `--ref` → Base.

## Always verify — you cannot hear these failures in a file listing

```bash
# content, loudness, pace, runaway length
python verify_takes.py --dir out/ --text-file line.txt --expect-seconds 12 --json-out asr.json
```

Flags `TOO QUIET` (rms < 0.02), `CONTENT` (< 95% match), `RUNAWAY`. It pins Whisper to
`temperature=0.0, condition_on_previous_text=False` **on purpose**: under default
temperature-fallback the same file scored 100% on one pass and 58.8% on another, and an
8-second clip was once transcribed as 25 repetitions of the sentence. Every "failure"
seen under default decoding turned out to be an ASR artifact.

```bash
# speaker similarity, in the FireRedTTS3 venv (CAM++ + its VoxCeleb ckpt ship there)
cd $VALKA_ROOT/FireRedTTS3
PYTHONPATH=$PWD .venv/bin/python <skill>/speaker_sim.py --takes out/ --refs refs.json
```

Take filenames must end `_<refid>.wav` to be matched to their reference.
**Never read the cosine without the chance floor it prints.** Speakers from one studio
already resemble each other: 11 live-dealer references scored **0.253 mean** cosine
against *each other* (max 0.585), so 0.4 is not "cloned successfully" — the `margin`
column is the real signal. The floor is a property of the reference set: those same
clips after cleaning scored 0.160. **Score every take against the *raw* references**,
even takes generated from cleaned ones, or the two arms are not comparable.

## Prep reference audio before believing any comparison

Field recordings are usually too quiet and mostly silence, which makes every engine look
bad — quiet output, dropped sentences, runaway length.

```bash
python prep_refs.py --list infer_list_audio.txt --root /repo/root --out refs_clean/ \
    --transcripts reference_transcripts.json      # optional, carried into the index
```

Drops sub −45 dBFS frames (60 ms padding), normalizes rms to 0.08, warns when < 3 s of
speech survives. Keep **both** sets and report the delta — that is the informative number.
Cleaning is not uniformly good: on the dealer set it bought FireRedTTS3 +21.9 pp content
accuracy and cut 21 s off its worst runaway, but *lowered* CAM++ similarity for
FireRedTTS3 (−0.026) and Fish S2 Pro (−0.057), because desilencing removes prosody cues.

Reference transcripts (needed by 4 of 5 engines) come from
`faster-whisper large-v3`; check them — one dealer clip transcribed as looped Latvian
garbage, and that nonsense then gets fed to the clone as `--ref-text`.

## Per-engine traps

- **qwen3tts** — no `seed`, sampled decoding (`do_sample=True, temperature=0.9`), so
  nothing is reproducible. `max_new_tokens` caps **one chunk, not the total**: 256 → 10.8 s,
  1024 → 81.8 s, and one noisy reference chained 8 chunks to **655.28 s** (= 8 × 81.84).
  Use `--max-seconds`. Its `instruct=` corrupts *CustomVoice* output (once 71 s of garbage)
  but is fine on VoiceDesign — route pace/style through VoiceDesign. CustomVoice has only
  4 female speakers and two of them (Vivian, Serena) reliably drop repeated sentences.
- **voxcpm2** — voice design goes in parentheses at the head of the text; the wording is
  fragile. "gentle and sweet" drove a female take to a near-whisper (rms 0.017) while still
  transcribing 100%; the backend appends `normal volume` when your `--instruct` omits it.
  With no reference, timbre *and* pace come from `seed` — pin it and listen.
- **fireredtts3** — highest speaker similarity of the five (0.629 vs floor 0.253) but the
  least stable length: 6 of 11 raw-reference takes ran past 15 s, worst 64 s. Needs
  flash-attn (hardcodes `flash_attention_2` in 3 places). The `[speed]` tag it echoes in
  `gen_text` is **not** the real pace — takes tagged `moderate` came out shorter than
  `very_fast` ones. It also has the only numeric control of the five:
  `generate_acoustic_edit("adjust the speed to 1.5x" | "shift the pitch by 3 steps")`,
  which is content- and pitch-preserving (verified: 0.5x→1.67× duration, pitch +3 semitones
  moved 266→313.8 Hz with duration unchanged), plus `generate_semantic_edit` to change words
  in existing audio.
- **fishs2pro** — 4B, ~17 GB VRAM, most accurate content (95.7%) and never ran long, but
  middling similarity (0.41). `generate_long` does `bool(prompt_tokens)`, so prompt args
  must be **lists**. The documented 3-step CLI reloads the 4B model each step; the backend
  drives the internals once instead.
- **indextts25** — the most reliable on noisy references: content 98.7–100%, never exceeded
  8.2 s, smallest similarity variance (±0.09), and **needs no reference transcript** — decisive
  when the transcripts themselves are unreliable. Extra knobs not exposed by `tts.sh`:
  `duration_factor` (0.5–2.0, >1 slows) and an 8-float `emo_vector`
  `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`.

## Measured baseline (11 dealer refs, one line, CAM++ floor 0.253)

Use as a sanity check when re-running; full detail in `valka-ai/tts_test_out/clone/README.md`.

| engine | spk-sim raw/clean | content raw/clean | worst duration |
|---|---|---|---|
| fireredtts3 | **0.629 / 0.602** | 61.0% / 82.9% | 64.0 s |
| indextts25 | 0.517 / 0.531 | 98.7% / **100%** | **8.2 s** |
| voxcpm2 | 0.417 / 0.442 | 62.4% / 70.1% | 20.3 s |
| fishs2pro | 0.410 / 0.354 | **95.7% / 96.8%** | 11.5 s |
| qwen3tts | 0.368 / 0.444 | 64.2% / 55.1% | **655.3 s** |

Pick: **indextts25** for reliability, **fireredtts3** for maximum similarity (budget for
reference cleaning + length guards), **fishs2pro** when the words must be exactly right.

## Install from scratch

Only if a `.venv` is missing. All five: `uv venv --python 3.12 .venv` (indextts25 and
fishs2pro need **3.11** — indextts requires `<3.12`), then:

```bash
# Qwen3-TTS / VoxCPM2 / IndexTTS-2.5 — plain editable install
uv pip install --python .venv/bin/python -e .

# FireRedTTS3 — requirements only (no pyproject → the package is never installed,
# which is why tts.sh puts the repo root on PYTHONPATH), and flash-attn is mandatory
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python flash-attn==2.8.3 --no-build-isolation

# Fish S2 Pro — two traps
#  1. pyaudio needs portaudio headers and inference never uses it → strip it from the deps
#  2. tensorboard 2.21 ships protobuf gencode newer than the pinned runtime, so
#     fish_speech.models.dac.modded_dac fails to import while hydra swallows the real
#     exception ("Error locating target ... DAC"). Pin tensorboard==2.18.0.
grep -v '^pyaudio' <(python - <<'P'
import re;print("\n".join(l.strip().rstrip(',').strip('"') for l in re.search(r'dependencies = \[(.*?)\n\]',open('pyproject.toml').read(),re.S).group(1).strip().split("\n") if l.strip()))
P
) > /tmp/fish_req.txt
uv pip install --python .venv/bin/python -r /tmp/fish_req.txt
uv pip install --python .venv/bin/python -e . --no-deps
uv pip install --python .venv/bin/python tensorboard==2.18.0
```

Weights: `hf download <model-id>` — Qwen3-TTS and VoxCPM2 use the HF cache; FireRedTTS3
(20 GB) → `FireRedTTS3/pretrained_models/`, Fish S2 Pro (11 GB) →
`FishS2/checkpoints/s2-pro/`, IndexTTS-2.5 (5.2 GB) → `IndexTTS-2.5/checkpoints/`.
Qwen3-TTS needs a separate checkpoint per control mode (`-CustomVoice`, `-VoiceDesign`, `-Base`).

If an `import` inside a backend resolves to the backend file itself, that is why the
backends are named `run_<engine>.py` — a file named `fireredtts3.py` shadows the package.
