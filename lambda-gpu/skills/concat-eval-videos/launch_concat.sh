#!/bin/bash
# Run concat_videos.py inside the singularity container on a GPU node (the only
# place imageio + an ffmpeg backend exist on this cluster). Blocks until done.
#
# Usage:
#   .claude/skills/concat-eval-videos/launch_concat.sh [--qos Q] [--time T] -- <args to concat_videos.py>
# Everything after `--` is forwarded verbatim to concat_videos.py. Example:
#   launch_concat.sh -- --runs output/runA output/runB \
#     --labels runA,runB --order-csv data/.../sample100.csv --out-dir output/compare_sample100
#
# Env/flag overrides: QOS (default boost_qos_dbg), TIME (00:30:00), ACCOUNT,
# PARTITION, CPUS, MEM, WORKSPACE (repo root, default cwd), SIF.
set -euo pipefail

QOS="${QOS:-boost_qos_dbg}"; TIME="${TIME:-00:30:00}"
ACCOUNT="${ACCOUNT:-aifac_f02_378}"; PARTITION="${PARTITION:-boost_usr_prod}"
CPUS="${CPUS:-8}"; MEM="${MEM:-64G}"
WORKSPACE="${WORKSPACE:-$(pwd)}"
WORK="${WORK:-/leonardo_work/AIFAC_F02_378}"
SHARED="$WORK/shared"
SIF="${SIF:-$SHARED/singularity/diffsynth-a100/diffsynth-bind-a100.sif}"

# Parse leading --qos/--time before the `--` separator (optional convenience).
while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --qos) QOS="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    *) echo "unknown launcher flag before --: $1" >&2; exit 2;;
  esac
done
[[ "${1:-}" == "--" ]] && shift || { echo "expected '--' before concat_videos.py args" >&2; exit 2; }

[[ -f "$SIF" ]] || { echo "SIF not found: $SIF (set SIF=...)" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -x
srun -A "$ACCOUNT" -p "$PARTITION" --qos "$QOS" \
  --gres=gpu:1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
  -J concat-videos \
  singularity exec --nv \
  -B "$WORKSPACE:/workspace" \
  -B "$SHARED:$SHARED" \
  -B "$SHARED/wan_models:/workspace/models" \
  -B "$WORKSPACE/outputs:/workspace/outputs" \
  --pwd /workspace \
  "$SIF" \
  python "${SCRIPT_DIR#$WORKSPACE/}/concat_videos.py" "$@"
