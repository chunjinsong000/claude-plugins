#!/usr/bin/env bash
# Dispatch one synthesis to the right engine venv. Each engine lives in its own
# venv (transformers 4.52-5.15 conflict), so this picks the interpreter for you.
#
#   tts.sh <engine> [--text ... | --text-file ...] --out out.wav [engine args]
#
# engines: qwen3tts voxcpm2 fireredtts3 fishs2pro indextts25
set -euo pipefail
VALKA="${VALKA_ROOT:-/home/ubuntu/chunjin/project/valka-ai}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

engine="${1:-}"; shift || true
case "$engine" in
  qwen3tts)    repo=Qwen3-TTS    ;;
  voxcpm2)     repo=VoxCPM2      ;;
  fireredtts3) repo=FireRedTTS3  ;;
  fishs2pro)   repo=FishS2       ;;
  indextts25)  repo=IndexTTS-2.5 ;;
  ""|-h|--help)
    sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "unknown engine: $engine" >&2; exit 2 ;;
esac

py="$VALKA/$repo/.venv/bin/python"
[ -x "$py" ] || { echo "missing venv: $py -- see SKILL.md 'Install from scratch'" >&2; exit 3; }

# Backends are named run_<engine>.py on purpose: a file called fireredtts3.py sits in
# sys.path[0] and shadows the fireredtts3 package it needs to import.
# cwd matters for indextts25 (relative checkpoints/) and keeps fish's hydra happy.
# FireRedTTS3 was never pip-installed as a package (it ships only requirements.txt), so
# its repo root must be on PYTHONPATH -- cd alone does NOT do it, because python puts the
# *script's* directory on sys.path[0], not the cwd.
cd "$VALKA/$repo"
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
     PYTHONPATH="$VALKA/$repo${PYTHONPATH:+:$PYTHONPATH}" \
     "$py" "$HERE/_backends/run_$engine.py" "$@"
