---
name: self-forcing
description: Launch, tune, and monitor the Self-Forcing experiments for Wan2.2-S2V-14B (task=sft/dmd_distill + --self_forcing) from live_dealer/train/self_forcing/. Picks the right launcher for the box (B200 SLURM+singularity vs bare-metal 2xH100), validates the NUM_FRAMES/block geometry before launch, applies knobs via env overrides, and knows the memory dials for each variant. Use when the user says "run/launch the self-forcing experiment", "start a self-forcing run", "sf training", "self-forcing SFT/DMD", "from-noise self-forcing", or asks why a self-forcing run OOMed or degenerated.
---

# Self-Forcing experiments (Wan2.2-S2V-14B live dealer)

Self-forcing trains the S2V model **block by block over its own streaming KV cache** —
block b+1 is conditioned on the state the model produced for block b, not on
ground-truth context — exactly as `livedealer_infer_real_streaming.py` runs at
deployment. This closes the train/inference drift gap that ordinary bidirectional SFT
never sees.

All launchers live in `live_dealer/train/self_forcing/` inside the DiffSynth repo.
There are three variants; pick by objective **and by box** (paths are hardcoded per box):

| Script | Task | Box | Block input | Supervision |
|--------|------|-----|-------------|-------------|
| `0_WanS2V_dmd_AVSpeech_selfforcing_3step_n25_fpb3.sh` | `sft --self_forcing` | **B200 node** (SLURM, partition `main`, 1x8 B200, singularity) | GT block noised at a random timestep | GT block (perceptual PFM/IV2 by default) |
| `1_WanS2V_dmd_selfforcing_3step_lambda.sh` | `dmd_distill --self_forcing` | 2xH100 lambda box (bare metal, `.venv`) | pure-noise rollout over the deployed few-step schedule | DMD gradient (teacher − critic score); no GT video loaded |
| `2_WanS2V_sft_selfforcing_fromnoise_2step_lambda.sh` | `sft --self_forcing --sf_from_noise` | **B200 node** (SLURM, same as 0_) | pure noise, fixed `--sf_timesteps` schedule, random end block AND random end step | GT block; `mse` default is the cheap baseline, **not** a phi limitation |

The `_lambda` suffix on `2_*` is a leftover from where it was first written; commit
357d030 ported it to the B200 node (SBATCH header, singularity, `/lambda/nfs/B200`
paths). Only `1_*` is still bare-metal.

Box-specific paths:
- B200 node (`150.136.211.158`): `WORK_DIR=/home/ubuntu/chunjin/DiffSynth`, shared FS
  `/lambda/nfs/B200` (SIF, `wan_models`, packed dataset), logs in
  `/home/ubuntu/chunjin/slurm_logs/`. 8x B200 183 GB, partition `main`, one job takes
  the whole node, so runs are SERIAL — queued jobs wait for the current one.
- Lambda 2xH100 box: `WORK_DIR=/home/ubuntu/chunjin/project/valka-ai/DiffSynth`,
  models/data under the repo. Script `1_*` will fail its prerequisite gate if run on
  the wrong box — that gate (not CUDA) is the first thing to check on an instant exit.

## Launching

Every knob is an env override in front of the launch command — never edit the script
for a one-off change:

```bash
# B200, SLURM (variant 0):
sbatch live_dealer/train/self_forcing/0_WanS2V_dmd_AVSpeech_selfforcing_3step_n25_fpb3.sh
NUM_FRAMES=37 SF_LOSS_TYPE=mse sbatch <same script>          # with knobs
GPUS_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 bash <same script>    # single-GPU smoke test, no SLURM

# B200, SLURM (variant 2, from-noise):
SF_LOSS_TYPE=none SF_L1_WEIGHT=1.0 RUN_NAME=<name> \
  sbatch --job-name=<short> live_dealer/train/self_forcing/2_WanS2V_sft_selfforcing_fromnoise_2step_lambda.sh

# Lambda box, bare metal (variant 1 only):
bash live_dealer/train/self_forcing/1_WanS2V_dmd_selfforcing_3step_lambda.sh
FAKE_UPDATE_RATIO=1 bash <same>                              # ~4x faster DMD steps
NUM_PROCESSES=1 CUDA_VISIBLE_DEVICES=0 bash <any>            # single-GPU smoke test
SAVE_STEPS=2 bash <any>                                      # exercise the save path fast
```

`sbatch` propagates the caller's environment (`--export=ALL` default), so the
`VAR=x sbatch <script>` form works. Always pass `RUN_NAME` when varying a knob the
default name does not encode — the default is
`..._${HEIGHT}p_${SF_LOSS_TYPE}_sft10800`, which does **not** include
`SF_L1_WEIGHT`, so two runs differing only in L1 weight would share an output dir and
overwrite each other's checkpoints. `--job-name=` on the sbatch line also renames the
log files (`%x_%j.out`), which is the only way to tell concurrent runs apart in
`slurm_logs/`.

Common knobs (all three): `NUM_FRAMES NUM_FRAME_PER_BLOCK SKIP_FRAMES FPS SF_LOSS_TYPE
SF_L1_WEIGHT LEARNING_RATE SAVE_STEPS RUN_NAME USE_STATIC_OBJ` and the debug dials
`SF_DEBUG SF_DEBUG_MAX SF_DEBUG_FPS SF_DEBUG_SUBDIR` (per-block `[GT | pred]` mp4s,
off by default). Variant-specific: `SF_SCORED_BLOCK ROPE_OFFSET` (0/2),
`DMD_TIMESTEPS CFG_SCALE FAKE_UPDATE_RATIO SF_GRAD_LAST_N HEIGHT WIDTH` (1),
`SF_TIMESTEPS HEIGHT WIDTH KV_CACHE_FRAMES` (2). Variant 0 has **no** `HEIGHT`/`WIDTH`
knob — it passes `--height 720 --width 1280` literally.

`SF_LOSS_TYPE` takes `perceptual | mse | both | none`:

| value | terms | phi loaded | needs a differentiable decode |
|-------|-------|-----------|-------------------------------|
| `perceptual` | phi/IV2 distance on decoded pixels | yes | yes |
| `mse` | latent MSE only | no | **no** |
| `both` | phi + latent MSE | yes | yes |
| `none` | nothing — `SF_L1_WEIGHT` is the only term | no | yes |

`SF_L1_WEIGHT>0` adds `w * mean|pixels_pred - gt_pixels|` at FULL decode resolution on
top of whichever of the above is selected. It is the only term that can see
card-scale detail (phi resizes every frame to 224x224, which puts a card under one
14px patch token). `none` requires `SF_L1_WEIGHT>0` and is rejected otherwise, by both
the launcher preflight and the loss function. With `none` the weight is redundant with
the learning rate — keep `SF_L1_WEIGHT=1.0` and tune `LEARNING_RATE`.

## Hard constraints — validate BEFORE launching

1. **Block geometry**: `NUM_FRAMES` must be `4 * NUM_FRAME_PER_BLOCK * k + 1`
   (= `12m+1` at fpb=3), AND yield **≥ 2 blocks** — with 1 block there is no
   autoregressive history and the run silently degenerates to per-block SFT
   (variant 0 warns; 1/2 exit). At fpb=3: 25→2 blocks, 37→3, 49→4, 289→24 (max the
   ~300-frame clips supply).
2. **`BATCH_SIZE` must stay 1** — `--self_forcing` rejects anything larger.
3. `SF_LOSS_TYPE=perceptual|both` needs the InternVideo2 phi weights
   (`internvideo2-stage2-1b/InternVideo2-stage2_1b-224p-f4.pt`). Staged on the B200
   shared FS, so all four loss types run there. Not on the lambda box.
4. `USE_STATIC_OBJ=true` needs the `static_obj/` jsonl tree next to the metadata CSV
   (the scripts preflight the first 50 rows and fail fast). On the lambda box only the
   11 test-set jsonls are staged, so it defaults off there.
5. `WAN_COMPILE_TRAINING` stays `false`: torch.compile does not compose with gradient
   checkpointing on the scored block (Dynamo guards on grad_mode; every rank dies in
   the first backward).
6. **`--sf_decode_context` and `--sf_grad_step` NO LONGER EXIST.** Commit ea4b275
   deleted both from the argparse without a replacement; `train.py` uses
   `parse_args()`, so passing either aborts with "unrecognized arguments" before step 1.
   The decode is fixed to `stream` and the gradient is hard-wired to the last step that
   ran. Do not resurrect the launcher flags without also restoring the plumbing.

### Two traps that kill a job in seconds (check these first on an instant exit)

- **`${BASH_SOURCE[0]}` under sbatch.** SLURM copies the launcher to
  `/var/spool/slurmd/jobNNNNN/slurm_script` and runs *that*, so `dirname` resolves to
  the spool dir and any `source "$(dirname "${BASH_SOURCE[0]}")/..."` fails — with
  `set -e` that kills the job before it allocates a single GPU (and `basename` records
  the launcher as "slurm_script"). Anchor on `$WORK_DIR` instead. This bit all three
  jobs of the 2026-08-05 batch; harmless when the same script is run bare with `bash`,
  which is why it can sit unnoticed.
- **The prerequisite gate.** Missing IV2 weights / static_obj tree / metadata / resume
  ckpt all exit with `[ERROR] missing prerequisites:` before training. Read the `.err`
  file, not the `.out`.

## Memory model — which dial to reach for on OOM

**The dominant cost is NOT the KV cache and NOT phi — it is the differentiable VAE
decode of the scored block.** Measured 2026-08-05 with `saved_tensors_hooks` on the
real modules (CPU, small resolution, extrapolated by area to 720x1280 / 12 frames):

| component | retained with grad | note |
|-----------|-------------------|------|
| **VAE decode of the scored block** | **~150 GiB** (874 saved tensors) | active whenever ANY pixel term is on (`perceptual`/`both`/`none`/`L1>0`) |
| same decode under `no_grad` | ~4 GiB (live set ~3 feature maps) | 39x less — this is what unscored blocks cost |
| phi (InternVideo2) activations | 0.87 GiB | all 40 blocks checkpointed, fixed 224x224 input |
| phi weights | 4.1 GiB | 1.03B params, fp32 on purpose |
| DiT bf16 weights | 28 GB | |
| rolling KV, **per denoising step** | 8.85 GB @720p | `2(k+v) * 40 layers * KV_CACHE_FRAMES * tokens_per_frame * 5120 * 2B` |
| shared cond cache (all steps) | 2.95 GB @720p | `2 * 40 * tokens_per_frame * 5120 * 2B` — the `_sf_init_kv_cache` docstring's "~0.4 GB" is wrong, and the `2_` header's cache table omits it entirely |

tokens_per_frame: 3600 @ 720x1280, 2304 @ 576x1024, 1296 @ 432x768.

Consequences, in order of how often they get this wrong:

- **Switching `perceptual` -> `none` saves ~5 GB, almost all of it phi's fp32 WEIGHTS,
  not its gradient.** It does not avoid the decode, which L1 needs too. Do not expect a
  pixel-L1-only run to fit somewhere a perceptual run does not.
- **`mse` with `SF_L1_WEIGHT=0` is the only configuration with no differentiable
  decode** (`grad_pixels = loss_type != "mse" or l1_weight > 0`), and is therefore
  dramatically cheaper than all the others — roughly 60-90 GB vs ~178 GB at 720p.
  Note it still runs one `no_grad` decode per block purely to walk the decoder's conv
  cache, which nothing then reads: ~20-30% of its step time is avoidable work.
- **Do NOT predict OOM by adding a delta to an observed `nvidia-smi` number.** That
  number is the caching allocator's high-water mark, not a hard requirement, and it is
  not additive. Measured: the 1-cache path (variant 0) and the 2-cache from-noise path
  (variant 2, `SF_TIMESTEPS=1000,556`, +8.85 GB of cache on the `end_step=1` steps)
  BOTH settle at **177.6 GB / 183.4 GB** at 720p and both run fine. An arithmetic
  prediction of 187 GB "must OOM" was wrong. Extrapolating the component table
  overshoots the real total by ~7%, i.e. +-13 GB — larger than any margin worth
  arguing about. **Settle it by running 4-5 steps, not by arithmetic.**
- Rough resolution scaling for everything except the 28 GB of weights: linear in pixel
  area (576x1024 = 0.64x, 432x768 = 0.36x of 720p).

- **Variant 0 (SFT, GT-noised)**: caches are per *block*; every scored block also pins
  a private KV-history copy (~8.8 GB at 720p/fpb3). `SF_SCORED_BLOCK=random` (default)
  is what makes 24 blocks affordable — `all` will not fit. Dials, in order:
  `NUM_FRAMES` (fewer blocks) → `NUM_FRAME_PER_BLOCK` → resolution.
- **Variant 2 (from-noise)**: `n_caches = end_step + 1`, drawn per step, so half the
  steps at a 2-entry schedule allocate two cache sets. Measured to fit at 720p anyway
  (see above). `NUM_FRAMES` is not a memory dial here.
- **Variants 1/2 (rollout)**: caches are per *denoising step*, NOT per block, and for
  DMD `n_caches = max(exit_flag)+1` — a step drawing the deepest exit holds one cache
  per schedule entry, so size for the worst case, not the average (720p measured
  77.2 GiB → dies on an 80 GB H100; 576x1024 is the DMD default, 432x768 the
  from-noise default). Dials, in order: `HEIGHT/WIDTH` → fewer schedule steps
  (`DMD_TIMESTEPS`/`SF_TIMESTEPS`) → `SF_GRAD_LAST_N=1`. **`NUM_FRAMES` is NOT a
  memory dial here** — it only changes how many no-grad prefix blocks each step walks.

DMD step time is dominated by `FAKE_UPDATE_RATIO` (~37 s per extra critic update
measured on 2xH100: 55 s/step at 1 vs 205 s/step at 5). Drop it to 1 for iteration.

## Checkpoints and resume

- Variants 0/2 (`task=sft`) resume via `FINE_TUNE_LORA=<step-N.safetensors>` — this
  continues THE SAME LoRA plus the ckpt's non-LoRA modules (`card_class_embedding` /
  `static_obj_embedding`, kept trainable). Default start: the PFM+static SFT
  checkpoint `..._SF150_pfm_iv2_SF25/step-10800.safetensors`.
- Variant 1 (`dmd_distill`) instead **merges** the SFT ckpt into the base
  (`TEACHER_CKPT`, `--merge_lora_to_base`) as the frozen teacher, then trains fresh
  student+critic LoRAs.
- **DMD deployment gotcha**: the saved student checkpoint holds LoRA tensors ONLY
  (800 keys, no embedding modules). Inference must load BOTH the SFT checkpoint (LoRA
  + embeddings) and the student LoRA; `live_dealer/infer/utils/cli.py --lora_path` is
  still single-valued — widen to `nargs="+"` before evaluating a student ckpt, or the
  card conditioning runs on random weights.
- Outputs: variant 0 → `$WORK_DIR/outputs/models/$RUN_NAME/` (bound as `/output`),
  variants 1/2 → `$WORK_DIR/output/models/$RUN_NAME/`. Per-step loss goes to
  `<run>/loss.csv` and TensorBoard.

## Run record: the from-noise loss-ablation batch (submitted 2026-08-05)

Four variant-2 runs, identical except for the loss, launched to answer *which
supervision actually carries card legibility when the block is generated from pure
noise*. Reproduce or extend with exactly these commands:

```bash
S=live_dealer/train/self_forcing/2_WanS2V_sft_selfforcing_fromnoise_2step_lambda.sh

# 83  perceptual + pixel L1 (w=1.0)  — the two pixel terms roughly balanced
SF_LOSS_TYPE=perceptual SF_L1_WEIGHT=1.0 HEIGHT=720 WIDTH=1280 \
RUN_NAME=2_WanS2V_sft_sf_fromnoise_n289_fpb3_720p_perceptual_l1w1.0_sft10800 \
  sbatch --job-name=sf_fn_perc_l1 "$S"

# 84  image-space L1 ONLY (needs the SF_LOSS_TYPE=none support)
SF_LOSS_TYPE=none SF_L1_WEIGHT=1.0 HEIGHT=720 WIDTH=1280 \
RUN_NAME=2_WanS2V_sft_sf_fromnoise_n289_fpb3_720p_l1only_w1.0_sft10800 \
  sbatch --job-name=sf_fn_l1only "$S"

# 85  latent-space MSE ONLY — the cheap baseline, and the only decode-free config
SF_LOSS_TYPE=mse SF_L1_WEIGHT=0 HEIGHT=720 WIDTH=1280 \
RUN_NAME=2_WanS2V_sft_sf_fromnoise_n289_fpb3_720p_mseonly_sft10800 \
  sbatch --job-name=sf_fn_mse "$S"

# 86  perceptual + pixel L1 (w=10.0) — L1-dominated
SF_LOSS_TYPE=perceptual SF_L1_WEIGHT=10.0 HEIGHT=720 WIDTH=1280 \
RUN_NAME=2_WanS2V_sft_sf_fromnoise_n289_fpb3_720p_perceptual_l1w10.0_sft10800 \
  sbatch --job-name=sf_fn_perc_l1w10 "$S"
```

Shared: 720x1280, `NUM_FRAMES=289` / `fpb=3` -> 24 blocks, `SF_TIMESTEPS=1000,556`,
`SF_SCORED_BLOCK=random`, `LR=1e-5`, `SAVE_STEPS=200`, `USE_STATIC_OBJ=true`, resumed
from `..._SF150_pfm_iv2_SF25/step-10800.safetensors`. One node each, so they run
serially; ~200 steps ≈ 4.5 h.

Measured throughput at 720p: **~70-87 s/step** (variant 2, perceptual+L1), against
**64 s/step** for variant 0 at the same resolution. The from-noise path runs ~1.5x the
DiT forwards (avg 12.5 blocks x avg 1.5 chain steps) but ONE FEWER decode per step
(12.5 vs 13.5): variant 0's unscored blocks decode the GT block to walk the conv cache,
and its scored block decodes twice (pred + target), while from-noise decodes only each
block's own prediction. Net effect on peak memory: none (the removed decode is
`no_grad` and does not occur at the peak moment).

### Reading these loss curves

- **`loss.csv` is 3-column: `step,loss,sf_end_step`.** The curve is a MIXTURE of one
  series per end step (`end_step=0` is a single jump from sigma 1 and sits above
  `end_step=1`). Split by the third column before comparing anything.
- **Absolute values are not comparable across loss types**, and not comparable with
  runs from before fd08327/ea4b275: the pixel target changed from `decode(GT latents)`
  to the REAL input frames, so the loss no longer approaches 0 — the VAE's own
  reconstruction error (measured 0.0144 pixel L1 / 35.4 dB at 432x768 on this data) is
  an irreducible floor.
- **A FLAT phi CURVE IS NOT EVIDENCE OF NO LEARNING.** Variant 0's perceptual loss sat
  at 0.059 -> 0.062 across 2119 steps, dead flat — and its checkpoints were then shown
  by eval to have measurably REDUCED the block-boundary jump. The phi distance is not a
  usable progress signal on this data; judge by eval renders, and do not recommend
  killing a run because its loss looks flat (that call was made, and was wrong).
- Observed scales: variant 0 perceptual **0.059** (flat, see above); variant 2
  perceptual+L1 w=1 starts at **0.12-0.16** and DOES descend — split by end step, both
  series fell ~9-11% over the first 116 steps (end_step=0: 0.1249 -> 0.1138,
  end_step=1: 0.1147 -> 0.1022). The L1 term is what makes the curve readable. The two
  end-step series sit only ~9% apart, much closer than "far above" would suggest.
- Neither `loss.csv` nor TensorBoard splits the phi and L1 components. To recover the
  L1 magnitude, difference two runs that share an init and differ only by the L1 term:
  `mean(job 83) - mean(job 84)` over the first ~50 steps ≈ `w * E[L1]`.

- B200 SLURM run: `squeue`, then `/home/ubuntu/chunjin/slurm_logs/<job-name>_<jobid>.{out,err}`.
  Emails on BEGIN/END/FAIL.
- Bare-metal runs print `[Step N] ...` losses to stdout (`PYTHONUNBUFFERED=1` is set —
  if a redirected run looks silent for ~130 steps, that export got lost).
- Visual check: relaunch/short-run with `SF_DEBUG=true` for per-block `[GT | pred]`
  mp4s under `<run>/<SF_DEBUG_SUBDIR>/` (caps at `SF_DEBUG_MAX` **per rank**).

## Why this is being trained: the 12-frame block-seam jump

**The point of these runs is temporal dependency, not spatial detail.** The symptom
they target: at test time the streaming output jumps every **12 pixel frames** — which
is exactly one block (3 latent frames at fpb=3). Established so far:

- The discontinuity is **DiT-side**, not the VAE (the latents themselves jump; a
  bidirectional decode of the same latents does not remove it). Both the DiT block and
  the VAE decode chunk are 3 latent frames, so always separate the two before acting:
  check whether `|z[t] - z[t-1]|` spikes at `t % 3 == 0`, or decode one latent sequence
  both streaming and bidirectionally and compare.
- **Self-forcing demonstrably reduces the jump** — confirmed by eval on variant 0's
  checkpoints (the RECONSTRUCTIVE variant, GT-block-noised + perceptual). So the win
  comes from the self-forcing *structure* — training each block over the model's own KV
  cache — not from the from-noise objective, which is a separate thing layered on top.

Why a seam survives at all: it is **unsupervised in all three directions**.

1. The loss only ever scores ONE block (`b_star`) against its own GT, and every term
   (phi, L1, MSE) is computed strictly INSIDE that block's 12 frames. Nothing compares
   block b's first frame to block b-1's last frame.
2. The KV handoff is **detached** at SelfAttention's write site, so block b's loss sends
   no gradient into block b-1's output — the model is never told to produce something
   the next block can continue from.
3. The rollout **stops** at `b_star`; nothing runs after the scored block.

And per-frame losses are nearly indifferent to the failure mode: an error that is
roughly CONSTANT within a block (a slight exposure or colour offset, say) is cheap under
L1/MSE frame by frame, yet reads as a hard cut at the boundary. Note also that **L1 and
latent MSE have no temporal structure at all** — phi (a video ViT with joint
spatio-temporal attention) is the only existing term that can see motion, so a
`SF_LOSS_TYPE=mse` run carries zero temporal supervision.

Levers, in cost order:

1. **A seam term.** The previous block was already decoded under `no_grad` to walk the
   conv cache, so its last pixel frame is free: add
   `|(pred_b[0] - prev_last) - (gt_b[0] - gt_prev_last)|`, plus an in-block temporal
   difference `|delta_pred - delta_gt|` over all 12 frames (supervises motion rather
   than content, which is exactly the gap above). `prev_last` is a constant, so gradient
   flows only into the new block — the intended direction.
2. **Widen the direct attention window**, `KV_CACHE_FRAMES` 3 -> 6/9. Today
   `sf_kv_cache_frames` defaults to `num_frame_per_block`, and when
   `cache_len == seg_len_block` the cache is entirely REPLACED by the current block
   (`wan_video_dit.py` ~484-490) — so block b attends directly to block b-1 and nothing
   else; earlier context survives only transitively through hidden states. This is the
   structural ceiling on temporal dependency. Costs +8.85 GB per denoising step per +3
   latent frames at 720p (x2 steps = +17.7 GB for 3->6), against ~5.8 GB of headroom —
   **blocked until the decode is made cheaper** (per-frame checkpointing frees ~80 GiB;
   see the memory section). Inference `--lframes_per_kv_cache` must move with it.
3. Let gradient cross the seam (score two consecutive blocks, or stop detaching one step
   back). Expensive — the detach is what keeps the per-block graphs independent and the
   checkpointed recompute correct.
4. Correlate the per-block noise instead of drawing i.i.d. `randn` per block. Whatever
   the conditioning does not determine is redrawn every block, which is itself a jump
   source. Must change on the inference side too.

## Inference-parity invariants (do not "fix" these)

These match deployment (`livedealer_infer_real_streaming.py`) on purpose:
`NUM_FRAME_PER_BLOCK=3` == `--lframes_per_block`, `SF_KV_CACHE_FRAMES` ==
`--lframes_per_kv_cache`, `ROPE_OFFSET=relative`, the per-block decode (fixed to
`stream` since ea4b275 removed the knob), `SF_SAMPLER=flow`, and the few-step
schedules must equal the `PARTITION_PRESETS`
entry the streaming run uses (`diffsynth/pipelines/pipeline_partition.py`:
3-step → `1000,768,358`, 2-step → `1000,556`). Known, deliberate divergences from
deployment: dense KV (no foreground pruning) and i.i.d. per-block noise.

## Procedure for the agent

1. Determine which box you are on (`hostname` / check which `WORK_DIR` exists) and
   which objective the user wants; that selects the script.
2. Check the geometry constraint for any `NUM_FRAMES`/`NUM_FRAME_PER_BLOCK` override,
   and the phi/static-obj prerequisites for the requested loss/conditioning.
3. Compose the launch as env overrides + `sbatch`/`bash` (see Launching), with an
   explicit `RUN_NAME` and `--job-name`. For a new config, run the single-GPU smoke
   test or a `SAVE_STEPS=2` short run first when the user is not explicitly launching a
   long production run.
4. **Watch the first ~5 steps before reporting success.** On the B200 that is ~10 min
   of model load plus ~5 steps; a launcher bug, a prerequisite miss, or an OOM all show
   up there and nowhere later. Poll `squeue -j <id>`, the `.err` file for
   "out of memory", `nvidia-smi` for the peak, and `<run>/loss.csv` for the first rows.
   A job that vanishes from `squeue` within seconds never reached CUDA — read `.err`.
5. Report the run name, output dir, peak memory, s/step, and the first loss values with
   their `sf_end_step`.

Do not predict OOM analytically and act on it — the 2026-08-05 batch was nearly
downscaled to 576x1024 on an arithmetic prediction that turned out to be wrong by the
whole margin. State the estimate, then measure.
