#!/usr/bin/env python3
"""Generate a livedealer_infer.py evaluation snippet for a training run.

Reads a training `*_leo_<N>.sh` script and emits the inference command that
evaluates it on the eyes-only test set, mirroring the canonical evals at the
bottom of live_dealer/infer/infer_card_class.sh:

  HF_HUB_OFFLINE=1 <WAN_* flags from training> python live_dealer/infer/livedealer_infer.py \
    --lora_path outputs/<run>/step-<STEP>.safetensors \
    --pose_video   <test_set>/infer_list_shape_body_only_eye.txt \  # if s2v_pose_video trained
    --object_video <test_set>/infer_list_shape_card_only.txt     \  # if s2v_object_video trained
    --input_image  <test_set>/infer_list_input_image.txt \
    --audio_path   <test_set>/infer_list_audio.txt \
    --gt_path      <test_set>/infer_list_gt.txt \
    --save_path output/<run>-<STEP> \
    --infer_frames 12 --num_clips 1 [--width W --height H] \
    --no_motion_video --use_block_attn

What it derives from the training script:
  * run name        <- basename of --output_path
  * WAN_* env flags  <- the tokens preceding `accelerate launch`
  * --width/--height <- training --width/--height
  * which inputs     <- training --extra_inputs (pose/object gating)
  * checkpoint step  <- latest step-*.safetensors (or --step)
"""
import argparse
import getpass
import glob
import os
import re
import shlex
import subprocess
import sys

TEST_SET = "data/project21_snapshot_12032025_packed/test_set"
DATA_ROOT = os.path.dirname(TEST_SET)  # base dir the data/ symlink points into
SLURM_LOGS_DIR = "/leonardo_scratch/large/userexternal/{user}/slurm_logs"

# Canonical 100-sample eyes-only metadata CSVs (see --csv). The pipeline reads the
# `card_detection` column only when present, so card-detection models (card_detection
# in --extra_inputs / WAN_CARD_CLASS_EMBED) want the one WITH that column, while
# card-encoder models (s2v_object_video, no card_detection) want the no_carddet one.
CSV_CARDDET = f"{TEST_SET}/metadata_12032025_v2_shape_S2VObject_new_eyes_only_leo_cards_sample100.csv"
CSV_NO_CARDDET = f"{TEST_SET}/metadata_12032025_v2_shape_S2VObject_new_eyes_only_sample100_no_carddet.csv"
AUTO_CSV = "auto"  # sentinel for bare `--csv` (pick CARDDET vs NO_CARDDET from config)
DEFAULT_EXTRA = "input_image,input_audio,s2v_pose_video,s2v_object_video"

# Canonical denoising schedule for distillation-model evals, matching
# direct_distill_loss() in diffsynth/pipelines/wan_video_new.py.
DISTILL_TIMESTEPS = "1000,768,358"

# Defaults for --launch, mirroring the interactive launch wrapper at the top of
# live_dealer/infer/infer_card_class.sh (srun + singularity exec --nv).
WORK_DEFAULT = "/leonardo_work/AIFAC_F02_378"
SIF_REL = "shared/singularity/diffsynth-a100/diffsynth-bind-a100.sif"
LAUNCH = dict(account="aifac_f02_493", partition="boost_usr_prod",
              qos="boost_qos_dbg", time="00:30:00", cpus="8", mem="64G")
AUTO_QOS = "auto"      # sentinel for --qos: pick by sample count (see SAMPLE_QOS_THRESHOLD)
SAMPLE_QOS_THRESHOLD = 10  # > this many samples -> non-preemptible `normal` QOS
NORMAL_QOS_TIME = "02:00:00"  # default wall time when auto-bumped to `normal`
LISTS = {
    "pose_video": f"{TEST_SET}/infer_list_shape_body_only_eye.txt",
    "object_video": f"{TEST_SET}/infer_list_shape_card_only.txt",
    "card_detection": f"{TEST_SET}/infer_list_card_detection.txt",
    "input_image": f"{TEST_SET}/infer_list_input_image.txt",
    "audio_path": f"{TEST_SET}/infer_list_audio.txt",
    "gt_path": f"{TEST_SET}/infer_list_gt.txt",
}


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def arg_value(text, flag):
    # Match the value on the SAME line as the flag and skip comment lines, so a
    # comment that merely mentions "--extra_inputs" at end-of-line doesn't make
    # us grab the first token of the next line.
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.search(rf"--{re.escape(flag)}[ \t]+(\S+)", line)
        if m:
            return m.group(1)
    return None


def resolve_base_from_jobid(job_id, logs_dir):
    """Map a SLURM job id back to the training *_leo_<N>.sh it ran.

    1. `scontrol show job <id>` -> `Command=<script>` (works while the job is
       running or still in the controller's memory).
    2. Fall back to the job's stdout log `<logs_dir>/*_<id>.out`, which echoes
       `The original script is located at: <script>` (works after the job is
       gone from the controller, as long as the log survives).
    """
    try:
        out = subprocess.run(
            ["scontrol", "show", "job", str(job_id)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True, timeout=15,
        )
        m = re.search(r"Command=(\S+)", out.stdout)
        if m and os.path.isfile(m.group(1)):
            return m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    user = os.environ.get("USER") or getpass.getuser()
    logdir = logs_dir.format(user=user)
    for log in sorted(glob.glob(os.path.join(logdir, f"*_{job_id}.out"))):
        try:
            with open(log) as f:
                for line in f:
                    m = re.search(r"original script is located at:\s*(\S+)", line)
                    if m and os.path.isfile(m.group(1)):
                        return m.group(1)
        except OSError:
            continue

    fail(f"could not resolve job {job_id} to a training script via scontrol or "
         f"{logdir}/*_{job_id}.out; pass --base instead")


def host_outputs(text, override):
    if override:
        return override
    m = re.search(r"^SCRATCH_PROJECT_DIR=(\S+)", text, re.MULTILINE)
    if not m:
        fail("could not find SCRATCH_PROJECT_DIR; pass --outputs-dir")
    return os.path.join(m.group(1).split("#")[0].strip(), "outputs")


def pick_step(ckpt_dir, want):
    if not os.path.isdir(ckpt_dir):
        fail(f"checkpoint dir does not exist: {ckpt_dir}")
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"step-(\d+)\.safetensors", f)
        if m:
            steps.append(int(m.group(1)))
    steps.sort()
    if not steps:
        fail(f"no step-*.safetensors in {ckpt_dir}")
    if want is None:
        return steps[-1], steps
    if want not in steps:
        fail(f"step-{want} not in {ckpt_dir} (have: {steps})")
    return want, steps


def wan_flags(text):
    """WAN_* env assignments that prefix the `accelerate launch` line."""
    m = re.search(r"^(.*)\baccelerate launch\b", text, re.MULTILINE)
    if not m:
        return []
    return re.findall(r"\bWAN_[A-Z0-9_]+=\S+", m.group(1))


def workspace_rel(path, workspace):
    """Rewrite an absolute CSV/data path to its workspace-relative form via the
    data/ symlink, so the snippet follows infer_card_class.sh's convention and
    resolves inside the container (which runs from /workspace). Relative paths
    pass through unchanged; paths outside the data/ tree warn and pass through."""
    if not os.path.isabs(path):
        return path
    data_real = os.path.realpath(os.path.join(workspace, DATA_ROOT))
    p = os.path.realpath(path)
    if p == data_real or p.startswith(data_real + os.sep):
        return os.path.join(DATA_ROOT, os.path.relpath(p, data_real))
    print(f"# WARN: {path} is not under {DATA_ROOT}; using the path as given "
          "(ensure it is bind-mounted in the container)", file=sys.stderr)
    return path


def csv_tag(csv):
    """A compact save_path/log suffix distinguishing a CSV eval from a list-file
    eval (and one CSV from another), e.g. 'sample100'."""
    stem = os.path.splitext(os.path.basename(csv))[0]
    # `sampleN` plus any trailing descriptors (e.g. `sample100_no_carddet`), so the
    # card-detection and no_carddet CSVs don't collide on the same save_path.
    m = re.search(r"sample\d+(?:_\w+)*", stem)
    if m:
        return m.group(0)
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:24] or "csv"


def count_test_samples(csv_abs, workspace):
    """Number of test items the eval will run: CSV data rows (minus header), or
    non-blank lines in the input_image list (the per-sample list, in list mode).
    Returns None if it can't be determined (caller falls back to the dbg QOS)."""
    try:
        if csv_abs:
            with open(csv_abs) as f:
                return max(0, sum(1 for _ in f) - 1)
        with open(os.path.join(workspace, LISTS["input_image"])) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def launch(snippet, run, step, args, tag=""):
    """Run the snippet on a GPU node via srun + singularity exec --nv.

    Mirrors the interactive launch wrapper at the top of infer_card_class.sh,
    converted to a non-interactive single-command srun. Output is teed to
    eval_logs/eval_<run>-<step>.log under the workspace. Blocks until the srun
    finishes; the caller (agent) typically runs this in the background.
    """
    workspace = args.workspace or os.getcwd()
    work = os.environ.get("WORK", WORK_DEFAULT)
    shared = os.path.join(work, "shared")
    sif = args.sif or os.path.join(work, SIF_REL)
    if not os.path.isfile(sif):
        fail(f"SIF not found: {sif} (pass --sif)")

    log_dir = os.path.join(workspace, "eval_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"eval_{run}-{step}{tag}.log")

    cmd = [
        "srun", "-A", args.account, "-p", args.partition, "--qos", args.qos,
        "--gres=gpu:1", "--ntasks=1", f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}", f"--time={args.time}",
        "singularity", "exec", "--nv",
        "-B", f"{workspace}:/workspace",
        "-B", f"{shared}:{shared}",
        "-B", f"{shared}/wan_models:/workspace/models",
        "-B", f"{workspace}/outputs:/workspace/outputs",
        "--pwd", "/workspace",
        sif,
        "/bin/bash", "-c", snippet,
    ]
    print(f"# launching on slurm: {args.qos} {args.time} 1xGPU; "
          f"log -> {log_path}", file=sys.stderr)
    print("# " + " ".join(shlex.quote(c) for c in cmd), file=sys.stderr)
    with open(log_path, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"# eval finished rc={rc}; full log -> {log_path}", file=sys.stderr)
    sys.exit(rc)


STREAM_SCRIPT = "live_dealer/infer/livedealer_infer_real_streaming.py"
STREAM_DEFAULT_WH = ("1280", "720")  # canonical 720p when training config has no width/height


def streaming_snippet(env, lora_path, extra, width, height, run, step, save_suffix, args):
    """Build a real-streaming eval snippet: a per-sample loop over the index-aligned
    eyes-only test lists, one torchrun invocation per clip.

    The streaming launcher (livedealer_infer_real_streaming.py) reads .txt lists for
    pose/card/audio/gt but only a SINGLE --input_image per run, so to keep each clip's
    dealer matched we iterate the lists with `paste ... | while read` and feed one row
    per invocation. WAN_CARD_CLASS_EMBED is auto-set by the launcher's CLI from
    --card_detection; cards are only supported via detections (card_video was removed).
    """
    w = width or STREAM_DEFAULT_WH[0]
    h = height or STREAM_DEFAULT_WH[1]
    if not width or not height:
        print(f"# WARN: no width/height from config; defaulting streaming to {w}x{h}",
              file=sys.stderr)

    has_card = "card_detection" in extra
    if not has_card and "s2v_object_video" in extra:
        print("# WARN: streaming dropped card_video support; a card-encoder (s2v_object_video) "
              "model gets NO card conditioning here. Only card_detection (WAN_CARD_CLASS_EMBED) "
              "models are card-conditioned in streaming.", file=sys.stderr)

    # paste columns / read vars, index-aligned; CARD only referenced when card-conditioned.
    cols = [LISTS["pose_video"], LISTS["card_detection"], LISTS["audio_path"],
            LISTS["input_image"], LISTS["gt_path"]]
    save_path = f"output/{run}-{step}{save_suffix}-stream"

    body = [
        f"paste -d '\\t' {' '.join(cols)} \\",
        "| while IFS=$'\\t' read -r POSE CARD AUDIO IMG GT; do",
        f"  {env} torchrun --standalone --nproc_per_node={args.gpus} \\",
        f"    {STREAM_SCRIPT} \\",
        f"    --lora_path {lora_path} \\",
        '    --input_image "$IMG" \\',
        '    --pose_video "$POSE" \\',
    ]
    if has_card:
        body.append('    --card_detection "$CARD" \\')
    body += [
        '    --audio_path "$AUDIO" \\',
        '    --gt_path "$GT" \\',
        f"    --save_path {save_path} \\",
        f"    --height {h} --width {w} --fps {args.fps} \\",
        f"    --num_clips {args.num_clips} --frames_per_clip {args.frames_per_clip} "
        f"--lframes_per_block {args.lframes_per_block} \\",
        f"    --num_inference_steps {args.num_inference_steps} --motion_frames {args.motion_frames} \\",
        f"    --rope_offset {args.rope_offset} --save_layout {args.save_layout}",
        "done",
    ]
    return "\n".join(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--base", help="path to training *_leo_<N>.sh")
    src.add_argument("--job-id", help="SLURM job id; resolves the training script "
                                      "via scontrol, then the job's slurm log")
    ap.add_argument("--checkpoint", "--lora", dest="checkpoint", default=None,
                    metavar="PATH",
                    help="evaluate a checkpoint path directly (e.g. "
                         "outputs/<run>/step-<N>.safetensors). The run name and step are "
                         "derived from the path, so the checkpoint dir need not match any "
                         "training script's --output_path. Combine with --base/--job-id to "
                         "pull the training config (WAN_* flags, resolution, extra_inputs), "
                         "or pass --wan-env/--extra-inputs/--width/--height to set it explicitly.")
    ap.add_argument("--wan-env", action="append", default=None, metavar="WAN_X=Y",
                    help="WAN_* env assignment(s) for inference; repeatable, or one "
                         "space/comma-separated string. Overrides those derived from --base.")
    ap.add_argument("--extra-inputs", dest="extra_inputs", default=None,
                    help="override the trained --extra_inputs gating "
                         "(e.g. 'input_image,input_audio,s2v_pose_video,card_detection')")
    ap.add_argument("--csv", "--dataset-metadata", dest="csv", nargs="?", const=AUTO_CSV,
                    default=None, metavar="PATH",
                    help="evaluate on a metadata CSV (--dataset_metadata_path) instead of the "
                         "hard-coded eyes-only list files. Bare --csv auto-picks the canonical "
                         "100-sample CSV from the config: the card-detection CSV when card_detection "
                         "is in --extra_inputs, else the no_carddet CSV (card-encoder models). Pass "
                         "an explicit PATH to force a specific CSV. Expected columns: video, "
                         "input_audio, s2v_pose_video, s2v_object_video, input_image[, card_detection]. "
                         "The CSV carries every per-input list, so the pose/object/card_detection "
                         "gating is bypassed (the pipeline uses what its WAN_* config needs). Absolute "
                         "paths under the data/ symlink are rewritten workspace-relative. Combine with "
                         "--base/--job-id/--checkpoint for the training config (WAN_* flags, resolution).")
    ap.add_argument("--dataset-base-path", dest="dataset_base_path", default=None, metavar="DIR",
                    help=f"base dir prepended to CSV-relative paths (default: {DATA_ROOT})")
    ap.add_argument("--save-tag", dest="save_tag", default=None,
                    help="override the auto-derived save_path/log suffix used in CSV mode "
                         "(default: a slug from the CSV filename, e.g. 'sample100')")
    ap.add_argument("--width", default=None, help="override inference width")
    ap.add_argument("--height", default=None, help="override inference height")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest); "
                    "ignored when --checkpoint is given (step comes from the path)")
    ap.add_argument("--outputs-dir", default=None)
    ap.add_argument("--slurm-logs-dir", default=SLURM_LOGS_DIR,
                    help="slurm log dir template ({user} expands to $USER) for --job-id")
    ap.add_argument("--append", action="store_true",
                    help="append the snippet to live_dealer/infer/infer_card_class.sh")
    ap.add_argument("--distill", nargs="?", const=DISTILL_TIMESTEPS, default=None,
                    metavar="T0,T1,...",
                    help="evaluate as a distillation model: denoise in a few fixed timesteps "
                         f"instead of the default schedule. Bare --distill uses {DISTILL_TIMESTEPS} "
                         "(the direct_distill_loss schedule); pass a comma-separated list to override. "
                         "Disables tea-cache and tags the save_path with -distill.")
    ap.add_argument("--launch", action="store_true",
                    help="run the eval on a GPU node via srun + singularity exec --nv "
                         "(non-interactive), teeing output to eval_logs/")
    # --- streaming mode (livedealer_infer_real_streaming.py) ---
    st = ap.add_argument_group("streaming mode (--streaming)")
    st.add_argument("--streaming", action="store_true",
                    help="emit the real-streaming (4-step causal, KV-cache) eval instead of "
                         "the bidirectional livedealer_infer.py command: a torchrun launch of "
                         "live_dealer/infer/livedealer_infer_real_streaming.py that generates the "
                         "FULL video (num_clips auto-caps to the pose length). Because the streaming "
                         "launcher takes a single --input_image per run, the snippet is a per-sample "
                         "loop over the index-aligned test lists (one invocation per clip, matched "
                         "dealer). Cards are fed via --card_detection only (card_video was removed); "
                         "incompatible with --csv (the launcher has no --dataset_metadata_path).")
    st.add_argument("--gpus", default="gpu",
                    help="torchrun --nproc_per_node for streaming (default: 'gpu' = all visible "
                         "GPUs; pass an int like 1 for single-GPU). The pipeline runs the same path "
                         "for 1/2/4/N ranks.")
    st.add_argument("--fps", type=int, default=30, help="streaming output fps (default 30)")
    st.add_argument("--frames-per-clip", dest="frames_per_clip", type=int, default=12,
                    help="streaming image frames per clip (default 12)")
    st.add_argument("--num-clips", dest="num_clips", type=int, default=100,
                    help="streaming clip cap; auto-reduced to the pose/audio length at runtime "
                         "(default 100 -> full video)")
    st.add_argument("--lframes-per-block", dest="lframes_per_block", type=int, default=3,
                    help="streaming latent frames per block (default 3)")
    st.add_argument("--num-inference-steps", dest="num_inference_steps", type=int, default=4,
                    help="streaming denoising steps (default 4; the causal/distill regime)")
    st.add_argument("--motion-frames", dest="motion_frames", type=int, default=9,
                    help="streaming motion frames (default 9)")
    st.add_argument("--rope-offset", dest="rope_offset", default="relative",
                    choices=["real", "relative"],
                    help="streaming RoPE offset mode (default relative = training convention)")
    st.add_argument("--save-layout", dest="save_layout", default="left_gt_right_gen",
                    choices=["top_gen_bottom_gt", "left_gt_right_gen", "left_pose_right_gen", "gen_only"],
                    help="streaming output composition (default left_gt_right_gen)")
    ap.add_argument("--workspace", default=None,
                    help="repo root to bind as /workspace (default: cwd)")
    ap.add_argument("--sif", default=None, help="override singularity image path")
    ap.add_argument("--account", default=LAUNCH["account"])
    ap.add_argument("--partition", default=LAUNCH["partition"])
    ap.add_argument("--qos", default=AUTO_QOS,
                    help=f"srun --qos (default: auto -> 'normal' when >{SAMPLE_QOS_THRESHOLD} "
                         f"samples, else '{LAUNCH['qos']}')")
    ap.add_argument("--time", default=LAUNCH["time"], help="srun --time (HH:MM:SS)")
    ap.add_argument("--cpus", default=LAUNCH["cpus"])
    ap.add_argument("--mem", default=LAUNCH["mem"])
    args = ap.parse_args()

    if not (args.base or args.job_id or args.checkpoint):
        ap.error("provide --checkpoint, --base, or --job-id")

    if args.streaming and args.csv is not None:
        ap.error("--streaming is incompatible with --csv: the streaming launcher reads "
                 "pose/card/audio/gt as .txt lists and has no --dataset_metadata_path.")
    if args.streaming and args.distill:
        print("# WARN: --distill has no effect in --streaming; streaming already denoises in "
              f"--num_inference_steps ({args.num_inference_steps}) fixed steps. Ignoring "
              "distill timesteps.", file=sys.stderr)

    # Config source (training script): optional when --checkpoint is given and the
    # config is supplied via --wan-env / --extra-inputs / --width / --height.
    base = args.base
    if args.job_id:
        base = resolve_base_from_jobid(args.job_id, args.slurm_logs_dir)
        print(f"# job {args.job_id} -> {base}", file=sys.stderr)
    text = None
    if base:
        if not os.path.isfile(base):
            fail(f"training script not found: {base}")
        text = open(base).read()

    # Run name + step + lora_path.
    all_steps = None
    if args.checkpoint:
        ckpt = args.checkpoint
        m = re.search(r"step-(\d+)\.safetensors$", os.path.basename(ckpt))
        if not m:
            fail(f"--checkpoint must point at a step-<N>.safetensors file: {ckpt}")
        step = int(m.group(1))
        run = os.path.basename(os.path.dirname(ckpt.rstrip("/")))
        if not run:
            fail(f"could not derive a run name from --checkpoint parent dir: {ckpt}")
        check = ckpt if os.path.isabs(ckpt) else os.path.join(args.workspace or os.getcwd(), ckpt)
        if not os.path.isfile(check):
            fail(f"checkpoint not found: {check}")
        lora_path = ckpt
    else:
        out_path = arg_value(text, "output_path")
        if not out_path:
            fail("no --output_path in training script")
        run = os.path.basename(out_path.rstrip("/"))
        step, all_steps = pick_step(
            os.path.join(host_outputs(text, args.outputs_dir), run), args.step)
        lora_path = f"outputs/{run}/step-{step}.safetensors"

    # Config: explicit overrides win, then the training script, then defaults.
    if args.extra_inputs is not None:
        extra = args.extra_inputs
    elif text is not None:
        extra = arg_value(text, "extra_inputs") or DEFAULT_EXTRA
    else:
        extra = DEFAULT_EXTRA
        print("# WARN: no --base/--extra-inputs; defaulting --extra_inputs gating to "
              f"'{DEFAULT_EXTRA}'", file=sys.stderr)
    width = args.width if args.width is not None else (arg_value(text, "width") if text else None)
    height = args.height if args.height is not None else (arg_value(text, "height") if text else None)
    if args.wan_env:
        flags = [f for item in args.wan_env for f in re.split(r"[,\s]+", item.strip()) if f]
    elif text is not None:
        flags = wan_flags(text)
    else:
        flags = []
        print("# WARN: no --base/--wan-env; emitting no WAN_* flags", file=sys.stderr)

    # Distillation runs (`--task direct_distill`) freeze extra modules like
    # card_encoder and DON'T save them in the checkpoint (only the LoRA). At
    # inference card_encoder is created via .to_empty() and, left unloaded, NaNs
    # the forward pass -> all-black video. Load those frozen modules from the base
    # checkpoint the run resumed from (its --lora_checkpoint), converting the
    # container path (/output/<base>/...) to the host/workspace-relative form
    # (outputs/<base>/...) that inference uses.
    is_distill = bool(args.distill) or (text is not None and "direct_distill" in (arg_value(text, "task") or ""))
    lora_ckpt = arg_value(text, "lora_checkpoint") if text else None
    extra_module_ckpt = None
    if is_distill and lora_ckpt:
        extra_module_ckpt = re.sub(r"^/?output/", "outputs/", lora_ckpt)

    # Resolve which CSV to evaluate on (if any): bare --csv auto-picks from the
    # config (card_detection in --extra_inputs -> card-detection CSV, else the
    # no_carddet CSV for card-encoder models); an explicit --csv PATH forces one.
    csv_src = None
    csv_abs = None
    if args.csv is not None:
        if args.csv == AUTO_CSV:
            needs_carddet = "card_detection" in extra
            csv_src = CSV_CARDDET if needs_carddet else CSV_NO_CARDDET
            print(f"# auto-selected {'card-detection' if needs_carddet else 'no-card-detection'} "
                  f"CSV ({'card_detection in --extra_inputs' if needs_carddet else 'card-encoder model'}): "
                  f"{csv_src}", file=sys.stderr)
        else:
            csv_src = args.csv
        csv_abs = csv_src if os.path.isabs(csv_src) else os.path.join(args.workspace or os.getcwd(), csv_src)
        if not os.path.isfile(csv_abs):
            fail(f"CSV not found: {csv_abs}")

    env = " ".join(["HF_HUB_OFFLINE=1", *flags])
    if args.streaming:
        # Real-streaming eval: torchrun loop over the test lists (see streaming_snippet).
        # CSV is rejected earlier; distill timesteps don't apply (streaming is few-step).
        # Helper appends "-stream" to save_path; save_suffix here is the launch/append tag.
        snippet = streaming_snippet(env, lora_path, extra, width, height, run, step, "", args)
        save_suffix = "-stream"
    else:
        lines = [f"{env} python live_dealer/infer/livedealer_infer.py \\",
                 f"  --lora_path {lora_path} \\"]
        if extra_module_ckpt:
            lines.append(f"  --extra_module_ckpt_path {extra_module_ckpt} \\")
        if csv_src:
            # The CSV carries every per-input column; livedealer_infer.py builds all
            # lists from it (pose/object/gt/input_image/audio/card_detection) and the
            # pipeline uses whichever its WAN_* config needs, so the extra_inputs
            # gating that selects list flags is irrelevant here.
            csv_path = workspace_rel(csv_src, args.workspace or os.getcwd())
            lines.append(f"  --dataset_metadata_path {csv_path} \\")
            lines.append(f"  --dataset_base_path {args.dataset_base_path or DATA_ROOT} \\")
        else:
            if "s2v_pose_video" in extra:
                lines.append(f"  --pose_video {LISTS['pose_video']} \\")
            if "s2v_object_video" in extra:
                lines.append(f"  --object_video {LISTS['object_video']} \\")
            # WAN_CARD_CLASS_EMBED runs feed raw (x, y, class) detections instead of a
            # rendered object video; the WAN_CARD_CLASS_EMBED=true flag is already carried
            # over by wan_flags(), and livedealer_infer.py ignores object_video in this mode.
            if "card_detection" in extra:
                lines.append(f"  --card_detection {LISTS['card_detection']} \\")
            lines.append(f"  --input_image {LISTS['input_image']} \\")
            lines.append(f"  --audio_path {LISTS['audio_path']} \\")
            lines.append(f"  --gt_path {LISTS['gt_path']} \\")
        save_suffix = "-distill" if args.distill else ""
        if csv_src:
            save_suffix += "-" + (args.save_tag or csv_tag(csv_src))
        lines.append(f"  --save_path output/{run}-{step}{save_suffix} \\")
        lines.append("  --infer_frames 12 \\")
        lines.append("  --num_clips 1 \\")
        if width:
            lines.append(f"  --width {width} \\")
        if height:
            lines.append(f"  --height {height} \\")
        # Distillation models denoise in a few fixed steps; feed them explicitly and
        # disable tea-cache (a many-step skip heuristic that's meaningless here).
        if args.distill:
            lines.append(f"  --custom_timesteps {args.distill} \\")
            lines.append("  --no_tea_cache \\")
        lines.append("  --no_motion_video \\")
        lines.append("  --use_block_attn")
        snippet = "\n".join(lines)

    if all_steps:
        print(f"# checkpoints in {run}: {', '.join(f'step-{s}' for s in all_steps)}", file=sys.stderr)
    print(f"# evaluating {run} step-{step}\n", file=sys.stderr)
    print(snippet)

    if args.append:
        tgt = "live_dealer/infer/infer_card_class.sh"
        distill_note = f" (distill {args.distill})" if args.distill else ""
        with open(tgt, "a") as f:
            f.write(f"\n\n# eval {run} @ step-{step}{distill_note}\n  {snippet}\n")
        print(f"\n# appended to {tgt}", file=sys.stderr)

    if args.launch:
        # Auto-select QOS by sample count: small debug runs stay on the preemptible
        # dbg QOS; larger evals go to the non-preemptible `normal` QOS (and get a
        # longer default wall time so they don't hit the 30-min dbg ceiling).
        if args.qos == AUTO_QOS:
            n = count_test_samples(csv_abs, args.workspace or os.getcwd())
            if n is not None and n > SAMPLE_QOS_THRESHOLD:
                args.qos = "normal"
                if args.time == LAUNCH["time"]:
                    args.time = NORMAL_QOS_TIME
            else:
                args.qos = LAUNCH["qos"]
            print(f"# auto-selected QOS '{args.qos}' "
                  f"({'unknown' if n is None else n} samples; "
                  f">{SAMPLE_QOS_THRESHOLD} -> normal @ {NORMAL_QOS_TIME}, "
                  f"else {LAUNCH['qos']})", file=sys.stderr)
        launch(snippet, run, step, args, tag=save_suffix)


if __name__ == "__main__":
    main()
