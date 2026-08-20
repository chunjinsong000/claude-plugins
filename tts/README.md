# tts — TTS engine workflows

Run and evaluate the five TTS engines deployed side by side under
`VALKA_ROOT` (default `/home/ubuntu/chunjin/project/valka-ai`):
**Qwen3-TTS**, **VoxCPM2**, **FireRedTTS3**, **Fish Audio S2 Pro**, **IndexTTS-2.5**.

Each engine has its own venv (their transformers pins conflict), so the skill ships a
dispatcher that picks the right interpreter.

## Skill

- **`synthesize`** — one CLI across all five engines for zero-shot voice cloning,
  natural-language voice design and built-in speakers; plus reference-audio prep and the
  ASR + CAM++ verification harness (with a chance floor, because a bare cosine means nothing).

```bash
tts.sh indextts25 --text-file line.txt --ref ref.wav --out out.wav
python verify_takes.py --dir out/ --text-file line.txt --expect-seconds 12
```

## Files

| file | what |
|---|---|
| `tts.sh` | dispatcher: engine → venv → backend |
| `_backends/run_*.py` | one per engine, identical CLI |
| `prep_refs.py` | desilence + loudness-normalize reference audio |
| `verify_takes.py` | Whisper round-trip: content, loudness, pace, runaway length |
| `speaker_sim.py` | CAM++ speaker similarity **with chance floor** |

Measured baselines and the per-engine traps are in the skill's `SKILL.md`.
