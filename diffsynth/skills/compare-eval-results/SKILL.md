---
name: compare-eval-results
description: Generate an HTML visualization summary comparing two card-detection evaluation results against ground truth. Use when given two result folders (e.g. inference_results/<checkpoint>, each with a labels/ subdir from run_pipeline.sh) and asked to compare them or build the comparison page. Produces side-by-side wrong-detection frames (pred A | pred B | GT), wrong-card position dots + density heatmap, cards-per-frame histogram, wrong crops tiled by GT rank and by GT suit per method, and per-rank / per-suit error-rate charts + tables.
---

# Compare evaluation results

Builds the comparison HTML page from **two** card-eval result folders, comparing
both predictions against ground truth on the **first frame** of each clip.

## Input

Each result folder is an `inference_results/<checkpoint>` dir produced by
`run_pipeline.sh`, containing a `labels/` subdir of per-frame YOLO label files
named `<video>_<frameid>.txt`. The matching ground-truth labels and the
generated videos are located automatically (see defaults).

## How to run

Run the wrapper **from the card_eval project root** (cwd) so it can locate the
`make_viz_html.py` engine and the project `.venv`. The wrapper is stdlib-only and
runs the engine under the project `.venv` itself. Launch it with the project venv
python:

```bash
.venv/bin/python ~/.claude/skills/compare-eval-results/compare_results.py <result_a> <result_b>
```

Example:

```bash
.venv/bin/python ~/.claude/skills/compare-eval-results/compare_results.py \
  inference_results/leo_card_class_embed_16nodes_cont1486_SF200-1000-sample100 \
  inference_results/leo_new_eyes_only_16nodes_card_encoder_cont1000_SF50-1000-sample100
```

The project root is auto-located by searching cwd upward for `make_viz_html.py`;
pass `--project-root PATH` if running from elsewhere. (If `.venv` is missing, build
it with `uv pip install -r requirements.txt`, or pass `--python <interpreter>`.)

### Defaults (override only if the layout differs)

| What | Default | Flag |
|------|---------|------|
| GT labels | `<result_a parent>/gt/labels` | `--gt-labels PATH` |
| GT videos | `/leonardo_work/AIFAC_F02_378/shared/livedealer/video_cut` | `--gt-video-dir PATH` |
| Generated videos | `/leonardo_work/AIFAC_F02_378/wxu/DiffSynth/output/<result-basename>` | `--pred-video-root DIR`, or `--video-a/--video-b PATH` |
| Output dir | `<result_a parent>/compare_<A>__vs__<B>/` | `--out-dir PATH` |
| Display names | result folder basenames | `--name-a/--name-b` |
| Stacked-panel crop / size | `720 / 1280 / 720` | `--pred-panel-height/--pred-width/--pred-height` |
| First frame | `0` | `--frame-id` |
| Match threshold | `0.02` (normalized) | `--pose-thresh` |

## Output

`<out-dir>/index.html` plus image assets (`*.jpg`, `*.png`). Report the
`index.html` path; the user opens it in a browser / VS Code preview. The page
sections, in order:

1. Per-clip side-by-side wrong-detection frames (pred A | pred B | GT), wrong cards boxed blue.
2. Wrong-detection position distribution (dots) and density heatmap, per method.
3. Cards-per-frame histogram + mean wrong detections vs table busyness.
4. Wrong crops tiled by **GT rank**, per method, + per-rank error-rate chart & table.
5. Wrong crops tiled by **GT suit**, per method, + per-suit error-rate chart & table.

## Notes

- The visualization engine is `make_viz_html.py` in the project root; this wrapper
  derives the paths and calls it. The video list is auto-built from result A's labels.
- Error rate is GT-centric (1 − recall): a GT card counts as an error if it was
  missed or assigned the wrong rank/suit. Totals match the eval metrics CSVs.
- Requires the project `.venv` (cv2, numpy, matplotlib, scipy) and read access to
  the generated + GT videos for cropping/rendering.
