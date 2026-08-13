#!/bin/bash
# Download the weights needed for LTX-2.3 LipDub (video dubbing) into $LTX2_ROOT/models.
#
# Prereqs (ONE TIME):
#   1. Accept the HF gate on each page (logged into your account):
#        https://huggingface.co/Lightricks/LTX-2.3
#        https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub
#        https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
#   2. Authenticate:  hf auth login    (Read token)   or   export HF_TOKEN=hf_xxx
#
#   bash download_lipdub_models.sh
set -euo pipefail

LTX2_ROOT="${LTX2_ROOT:-/home/ubuntu/chunjin/project/valka-ai/LTX-2}"
# hf CLI: prefer one on PATH, else the DiffSynth venv where it's installed here.
HF="${HF_BIN:-$(command -v hf || echo /home/ubuntu/chunjin/project/valka-ai/DiffSynth/.venv/bin/hf)}"

cd "$LTX2_ROOT"
DEST_LTX=models/ltx-2.3
DEST_GEMMA=models/gemma-3-12b

if ! "$HF" auth whoami >/dev/null 2>&1; then
    echo "ERROR: not logged into Hugging Face. Run 'hf auth login' (or export HF_TOKEN) and accept the model gates first." >&2
    exit 1
fi

echo "==> LTX-2.3 distilled base checkpoint + spatial upscaler"
"$HF" download Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled-1.1.safetensors \
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    --local-dir "$DEST_LTX"

echo "==> LipDub IC-LoRA"
"$HF" download Lightricks/LTX-2.3-22b-IC-LoRA-LipDub \
    ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors \
    --local-dir "$DEST_LTX"

echo "==> Gemma-3 12B text encoder (QAT q4, unquantized) — all files"
"$HF" download google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local-dir "$DEST_GEMMA"

echo ""
echo "Done. Weights under $LTX2_ROOT/:"
echo "  $DEST_LTX/ltx-2.3-22b-distilled-1.1.safetensors"
echo "  $DEST_LTX/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
echo "  $DEST_LTX/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
echo "  $DEST_GEMMA/  (Gemma text encoder)"
