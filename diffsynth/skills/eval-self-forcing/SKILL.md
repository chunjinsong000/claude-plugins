---
name: eval-self-forcing
description: Evaluate a self-forcing (causal, few-step) S2V run against a bidirectional/SFT baseline eval of the same test set. Builds a GT | baseline | self-forcing 3-panel comparison video, a face-centred head-zoom comparison, and a temporal-consistency matrix (warp error, tOF, temporal LPIPS, static-background flicker, PSNR drift curves) with charts. Use when the user says "evaluate the self forcing results", "compare self forcing against the baseline", "concat the results with the baseline for visualization", "zoom the head for comparison", or "calculate the temporal consistency metrics".
---

# Evaluate a self-forcing run vs a baseline

Takes the per-clip mp4 dirs of **two evals of the same test set** and produces the
three things always wanted when judging a causal/self-forcing model:

1. `compare_full.mp4` — `GT | baseline | self-forcing`, native 1280x720 per panel (3840x720).
2. `compare_head.mp4` — the same three panels cropped to a per-clip, face-centred
   384x384 box and 2x upscaled (2304x768). Drift and identity loss show up in the face first.
3. `temporal_consistency_matrix.md` + `temporal_curves.png` + `per_clip_psnr.png` —
   the temporal metrics, GT included as the reference column.

## Input assumption (check this first)

Both eval dirs must hold per-clip mp4s produced with `--save_layout left_gt_right_gen`,
i.e. **2560x720 = [GT 1280x720 | generation 1280x720]**. Verify with:

```bash
ffprobe -v error -show_entries stream=width,height,nb_frames -of default=nw=1 <clip>.mp4
```

If a run used a different layout (e.g. vertically stacked), the panel crops in
`build_comparison.py` / `temporal_metrics.py` (`PANEL_W`, `PANEL_H`) need adjusting.

Also confirm the two runs share inference settings — compare their
`*_module_timing.txt` headers (`Clips`, `blocks/clip`, `frames/block`) and the
`--num_inference_steps` / resolution in the launch script. A metric table across
different step counts or resolutions is not a comparison.

**Only clips present in both dirs are used**, so a still-running eval is fine — but
prefer to wait for it, and say in the report how many clips the numbers cover.
Check for a live run before starting: `ps aux | grep livedealer_infer`.

## Run it

```bash
~/.claude/skills/eval-self-forcing/run_all.sh \
  --baseline output/<baseline_run>/step-<N>/<set>/<cfg> \
  --sf       output/self_forcing/<run>/step-<N>/<set>/<cfg> \
  --out      output/self_forcing/<compare_dir> \
  --label-a  "Baseline bidir-SFT step-10800" \
  --label-b  "Self-Forcing step-600"
```

Runtime on a 52-core box, 11 clips x 300 frames: videos ~1 min, metrics ~4 min flow
(11 workers) + ~1 min/clip LPIPS on GPU. **Launch it as a background Bash command.**
Flags: `--jobs N`, `--stride 2` (halves flow cost), `--no-lpips`, `--skip-videos`,
`--skip-metrics`.

The four steps also run standalone: `detect_heads.py`, `build_comparison.py`,
`temporal_metrics.py`, `report.py` (all `--help`).

## The metrics, and how to read them

`temporal_metrics.py` streams GT / A / B together, per clip, full-frame at 640x360 and
inside the head box at native 384x384.

| Metric | Direction |
|---|---|
| `psnr` vs GT | higher better |
| `tlp` temporal LPIPS `LPIPS(I_t, I_t+1)` | **closeness to GT** |
| `tdiff` `mean\|I_t+1 − I_t\|` | **closeness to GT** |
| `warp_self` warp error with the video's own Farnebäck flow, occlusion-masked | **closeness to GT** |
| `warp_gtflow` warp error using the **GT's** flow field | lower better |
| `tof` `mean‖f_gen − f_gt‖₂` (px) | lower better |
| `static` flicker on pixels with GT temporal std < 2/255 | **closeness to GT** |

**The "closeness to GT" ones are the trap.** A model that is over-smoothed or nearly
frozen scores a *lower* `warp_self` / `tdiff` / `tlp` than the ground truth, which
looks like a win if you rank by "lower is better". Always print the GT column and
score by `|method − GT|`; `report.py` does this and names the winner per row.

`warp_gtflow`, `tof` and `psnr` are the ones where lower/higher genuinely is better —
they are anchored to GT rather than self-referential.

## Reading the drift curves

`temporal_curves.png` plots PSNR and `warp_self` against time-in-clip, averaged over
clips, 15-frame moving average. This is the point of the exercise for a causal model:
a self-forcing run that has fixed exposure bias should keep its PSNR **flat** across
the 10 s rollout, while a run that still drifts sags to the right. The GT reference
line is drawn on the warp panels.

The y-scale deliberately excludes the first 0.3 s — the ref-image→frame-0 transition
spikes the warp error for every video, GT included, and would flatten everything else.

## Reporting to the user

Quote the matrix, then say plainly which model wins on **GT-anchored** metrics vs
**self-referential** ones, and whether the PSNR curve is flat or sagging. Note the
clip count. Include a rendered frame of both comparison videos (extract with ffmpeg
and Read it) so the visual claim is backed by something you looked at.

## Gotchas

- **Head boxes** come from a Haar cascade on the GT panel, median over 24 sampled
  frames, fixed 384 px side so the zoom factor is identical across clips. Verify them
  once by tiling one cropped frame per clip and looking at it — a missed face falls
  back to the frame centre and is logged.
- The GT dealer sits a few px right of the generated dealer in some clips; the same
  box is used for all three panels, so that offset is visible in the head zoom. It is
  a real spatial offset, not a crop bug.
- `cv2.setNumThreads(1)` is set on purpose — parallelism comes from one process per
  clip. Raising it oversubscribes.
- Audio is copied from the baseline clip (it is the GT audio), so lip-sync is judgeable
  in the comparison videos.
- Per-clip `full_NN_*.mp4` / `head_NN_*.mp4` are kept next to the concatenated files
  for spot checks.
