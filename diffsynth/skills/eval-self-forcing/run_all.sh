#!/usr/bin/env bash
# Full self-forcing evaluation: head boxes -> comparison videos -> temporal metrics -> report.
#
#   run_all.sh --baseline DIR --sf DIR --out DIR [--label-a STR] [--label-b STR]
#              [--jobs N] [--stride N] [--no-lpips] [--skip-videos] [--skip-metrics]
#
# DIRs are the per-clip mp4 dirs of two evals of the SAME test set (2560x720,
# left_gt_right_gen).  Everything lands in --out.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/home/ubuntu/chunjin/project/valka-ai/DiffSynth}"
PY="${PY:-$REPO/.venv/bin/python}"

BASELINE=""; SF=""; OUT=""; LA="Baseline"; LB="Self-Forcing"
JOBS=11; STRIDE=1; NOLPIPS=""; SKIP_VIDEOS=0; SKIP_METRICS=0; CLIPS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) BASELINE="$2"; shift 2;;
    --sf)       SF="$2"; shift 2;;
    --out)      OUT="$2"; shift 2;;
    --label-a)  LA="$2"; shift 2;;
    --label-b)  LB="$2"; shift 2;;
    --jobs)     JOBS="$2"; shift 2;;
    --stride)   STRIDE="$2"; shift 2;;
    --no-lpips) NOLPIPS="--no-lpips"; shift;;
    --skip-videos)  SKIP_VIDEOS=1; shift;;
    --skip-metrics) SKIP_METRICS=1; shift;;
    --clips)    shift; while [[ $# -gt 0 && "$1" != --* ]]; do CLIPS+=("$1"); shift; done;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
CLIP_ARG=(); [[ ${#CLIPS[@]} -gt 0 ]] && CLIP_ARG=(--clips "${CLIPS[@]}")
[[ -n "$BASELINE" && -n "$SF" && -n "$OUT" ]] || { echo "need --baseline --sf --out" >&2; exit 2; }

mkdir -p "$OUT"
BOXES="$OUT/head_boxes.json"

echo "== 1/4 head boxes (from the baseline GT panel)"
[[ -f "$BOXES" ]] || "$PY" "$SKILL_DIR/detect_heads.py" "$BASELINE" "$BOXES"

if [[ $SKIP_VIDEOS -eq 0 ]]; then
  echo "== 2/4 comparison videos (full frame + head zoom)"
  "$PY" "$SKILL_DIR/build_comparison.py" --a "$BASELINE" --b "$SF" --out "$OUT" \
      --label-a "$LA" --label-b "$LB" --head-boxes "$BOXES" --jobs 8 "${CLIP_ARG[@]}"
fi

if [[ $SKIP_METRICS -eq 0 ]]; then
  echo "== 3/4 temporal metrics"
  "$PY" "$SKILL_DIR/temporal_metrics.py" --a "$BASELINE" --b "$SF" --out "$OUT" \
      --label-a "$LA" --label-b "$LB" --head-boxes "$BOXES" \
      --jobs "$JOBS" --stride "$STRIDE" $NOLPIPS "${CLIP_ARG[@]}"

  echo "== 4/4 report"
  "$PY" "$SKILL_DIR/report.py" --metrics "$OUT"
fi

echo
echo "done -> $OUT"
ls -la "$OUT" | grep -E "compare_|\.md|\.png|\.csv" || true
