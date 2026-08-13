#!/bin/bash
# Run LTX-2.3 LipDub (video dubbing): re-dub an existing video's speech to new
# dialogue while preserving the speaker's appearance, scene, and motion.
#
# Prereqs:
#   1. bash download_lipdub_models.sh   (after accepting HF gates + hf auth login)
#   2. A REFERENCE VIDEO with an audio track (the footage to be re-dubbed).
#
# LipDub is a DISTILLED-checkpoint, single-IC-LoRA, two-stage pipeline. Frame
# count + fps come from the reference video automatically (count snapped to
# nearest 8k+1) — there is NO --num-frames/--frame-rate. --prompt is the NEW
# dialogue/description the mouth is dubbed to.
#
# Usage:
#   bash run_lipdub.sh <reference_video.mp4> "<new dialogue prompt>" [output.mp4]
#
# Env:
#   LTX2_ROOT             LTX-2 repo root (default /home/ubuntu/chunjin/project/valka-ai/LTX-2)
#   CUDA_VISIBLE_DEVICES  GPU to use (default 1)
#   LIPDUB_LOWVRAM=1      add --quantization fp8-cast --offload cpu

set -euo pipefail

REF_VIDEO="${1:?Usage: run_lipdub.sh <reference_video.mp4> \"<prompt>\" [output.mp4]}"
PROMPT="${2:?Provide the new-dialogue prompt as the 2nd argument}"
OUTPUT="${3:-output_lipdub.mp4}"

LTX2_ROOT="${LTX2_ROOT:-/home/ubuntu/chunjin/project/valka-ai/LTX-2}"
cd "$LTX2_ROOT"

MODELS=models/ltx-2.3
GEMMA=models/gemma-3-12b
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Resolve reference video to an absolute path (we cd into LTX2_ROOT above).
case "$REF_VIDEO" in /*) ;; *) REF_VIDEO="$OLDPWD/$REF_VIDEO";; esac

EXTRA=()
if [ "${LIPDUB_LOWVRAM:-0}" = "1" ]; then
    EXTRA+=(--quantization fp8-cast --offload cpu)
fi

uv run python -m ltx_pipelines.lipdub \
    --distilled-checkpoint-path "$MODELS/ltx-2.3-22b-distilled-1.1.safetensors" \
    --spatial-upsampler-path    "$MODELS/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
    --gemma-root                "$GEMMA" \
    --lora                      "$MODELS/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors" \
    --reference-video           "$REF_VIDEO" \
    --reference-strength        1.0 \
    --prompt                    "$PROMPT" \
    --seed                      42 \
    --output-path               "$OUTPUT" \
    "${EXTRA[@]}"

echo "LipDub output -> $LTX2_ROOT/$OUTPUT"
