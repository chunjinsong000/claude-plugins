#!/usr/bin/env python3
"""Generate a continuation SLURM script from an existing training run.

Given a base training script, this:
  1. reads the run's --output_path (the run that produced the checkpoints),
  2. resolves the host checkpoint dir and picks the latest step-*.safetensors
     (or a specific step via --step),
  3. writes a new `..._leo_<N+1>.sh` that resumes from that checkpoint with
     --lora_checkpoint + --skip_frames, an updated --output_path suffix
     (_cont<STEP>_SF<SF>) and a concise #SBATCH --job-name.

It only edits the launch args; everything else in the base script is preserved.
Use --submit to sbatch the result; by default it just writes the file and
prints the suggested sbatch command so you can review first.
"""
import argparse
import os
import re
import subprocess
import sys


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def parse_arg_value(text, flag):
    """Return the value passed to a long flag like `--output_path /foo \\`."""
    m = re.search(rf"--{re.escape(flag)}\s+(\S+)", text)
    return m.group(1) if m else None


def resolve_host_outputs(text, override):
    if override:
        return override
    # Convention in these templates: OUTPUT_DIR defaults to SCRATCH_PROJECT_DIR/outputs,
    # and the container binds OUTPUT_DIR -> /output.
    m = re.search(r"^SCRATCH_PROJECT_DIR=(\S+)", text, re.MULTILINE)
    if not m:
        fail("could not find SCRATCH_PROJECT_DIR in base script; pass --outputs-dir explicitly")
    scratch = m.group(1).split("#")[0].strip()
    return os.path.join(scratch, "outputs")


def latest_step(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        fail(f"checkpoint dir does not exist: {ckpt_dir}")
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.fullmatch(r"step-(\d+)\.safetensors", f)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        fail(f"no step-*.safetensors checkpoints found in {ckpt_dir}")
    return max(steps), sorted(steps)


def next_script_path(base_path):
    d, name = os.path.dirname(base_path), os.path.basename(base_path)
    m = re.match(r"(.*_leo_)(\d+)(\.sh)$", name)
    if not m:
        fail(f"base script name does not match *_leo_<N>.sh: {name}")
    prefix, num, ext = m.group(1), int(m.group(2)), m.group(3)
    n = num + 1
    while os.path.exists(os.path.join(d, f"{prefix}{n}{ext}")):
        n += 1
    return os.path.join(d, f"{prefix}{n}{ext}")


def short_jobname(run_name, total_step, sf):
    # e.g. leo_new_eyes_only_16nodes_card_sequential_rope_compression -> "rope_compression"
    # total_step is the cumulative step count from the first run (matches the run-dir suffix).
    tail = re.sub(r"_cont\d+_SF\d+$", "", run_name)
    tail = "_".join(tail.split("_")[-2:]) or tail
    return f"{tail}_sf{sf}_cont{total_step}"[:64]


# Bash prelude injected in --runtime-latest mode. Placeholders are substituted
# (not an f-string: the body is full of bash ${...}/$(...) that must survive).
_RUNTIME_PRELUDE = r'''
# --- Resolve checkpoint to resume from AT RUNTIME (continue-training) --------
# This job is typically queued on a SLURM dependency behind the source run; by
# the time it starts, that run has finished OR died. Rather than hardcode a
# step, pick the newest step-*.safetensors actually on disk at launch, so a run
# that crashed near the end still resumes from its latest checkpoint. The
# _cont<N>_SF suffix (and any steps the parent already accumulated) is derived
# from that real step so the run name stays truthful. Exits 1 if none exist.
SRC_RUN=@@RUN@@
SRC_HOST_DIR="$OUTPUT_DIR/$SRC_RUN"
LATEST_CKPT=$(ls -1 "$SRC_HOST_DIR"/step-*.safetensors 2>/dev/null \
    | sed -E 's|.*/step-([0-9]+)\.safetensors$|\1 &|' | sort -n | tail -n1 | cut -d' ' -f2-)
if [ -z "$LATEST_CKPT" ]; then
    echo "[continue] ERROR: no step-*.safetensors in $SRC_HOST_DIR -- nothing to resume from." >&2
    exit 1
fi
RESUME_STEP=$(basename "$LATEST_CKPT" | sed -E 's|step-([0-9]+)\.safetensors|\1|')
RESUME_TOTAL=$(( @@PREV@@ + RESUME_STEP ))
export RESUME_CKPT="@@ROOT@@/$SRC_RUN/step-${RESUME_STEP}.safetensors"
export RESUME_RUN_OUT="@@ROOT@@/@@STRIPPED@@_cont${RESUME_TOTAL}_SF@@SF@@"
echo "[continue] resuming from $RESUME_CKPT (skip_frames @@SF@@) -> output $RESUME_RUN_OUT"
'''


def apply_runtime_latest(text, run_name, stripped, prev_total, container_out_root, sf):
    """Rewrite a container base script to resolve the checkpoint at launch time.

    The source run's checkpoint dir need not exist yet -- the emitted script
    globs the newest step-*.safetensors when the SLURM job actually runs, then
    feeds it (and a truthful _cont<step>_SF suffix) into the launch via the
    RESUME_CKPT / RESUME_RUN_OUT container env vars. Targets the project's
    container template (srun + `singularity exec` + `"$SIF"`); fails clearly if
    those anchors are absent.
    """
    prelude = (_RUNTIME_PRELUDE
               .replace("@@RUN@@", run_name)
               .replace("@@STRIPPED@@", stripped)
               .replace("@@ROOT@@", container_out_root)
               .replace("@@PREV@@", str(prev_total))
               .replace("@@SF@@", str(sf)))
    new = text

    # 1) job-name (step unknown at gen time -> "latest")
    new = re.sub(r"(#SBATCH --job-name=)\S+(.*)",
                 lambda m: f"{m.group(1)}{short_jobname(run_name, 'latest', sf)}"
                           f"            # continue {run_name} @ latest checkpoint, skip_frames {sf}",
                 new, count=1)

    # 2) inject the host-side resolver just before the training launch.
    m = re.search(r"\n(# --- Real training[^\n]*\n)?srun\b", new)
    if not m:
        m = re.search(r"\nsingularity exec\b", new)
    if not m:
        fail("could not find an `srun`/`singularity exec` launch line to anchor the runtime resolver")
    new = new[:m.start()] + "\n" + prelude + new[m.start():]

    # 3) pass the resolved paths into the container (before the `"$SIF"` line).
    m = re.search(r"(\n)(\s*)\"\$SIF\"\s*\\", new)
    if not m:
        fail("could not find a `\"$SIF\"` line to attach --env RESUME_* (runtime-latest needs the container template)")
    ind = m.group(2)
    envs = (f'{m.group(1)}{ind}--env RESUME_CKPT="$RESUME_CKPT" \\'
            f'\n{ind}--env RESUME_RUN_OUT="$RESUME_RUN_OUT" \\'
            f'{m.group(1)}{ind}"$SIF" \\')
    new = new[:m.start()] + envs + new[m.end():]

    # 4) output_path -> the runtime-resolved dir
    new = re.sub(r"(--output_path\s+)\S+", r'\1"$RESUME_RUN_OUT"', new, count=1)
    # 5) drop any existing checkpoint/skip_frames lines so we don't duplicate
    new = re.sub(r"\n\s*--lora_checkpoint\s+\S+\s*\\", "", new)
    new = re.sub(r"\n\s*--skip_frames\s+\S+\s*\\", "", new)
    # 6) insert checkpoint + skip_frames (referencing the container env vars) after --lora_rank
    m = re.search(r"(\n)(\s*)--lora_rank\s+\S+\s*\\", new)
    if not m:
        fail("could not find a `--lora_rank` line to anchor the inserted args")
    indent = m.group(2)
    insert = (f"{m.group(0)}"
              f'\n{indent}--lora_checkpoint "$RESUME_CKPT" \\'
              f'\n{indent}--skip_frames {sf} \\')
    new = new[:m.start()] + insert + new[m.end():]
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="path to base training .sh script")
    ap.add_argument("--skip-frames", type=int, required=True, dest="sf")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to resume from (default: latest)")
    ap.add_argument("--outputs-dir", default=None,
                    help="host dir holding run output folders (default: derived from SCRATCH_PROJECT_DIR/outputs)")
    ap.add_argument("--out", default=None, help="output script path (default: next *_leo_<N+1>.sh)")
    ap.add_argument("--runtime-latest", action="store_true", dest="runtime_latest",
                    help="resolve the newest step-*.safetensors AT LAUNCH TIME instead of now. "
                         "Use when the source run is still training (this job is queued on a "
                         "dependency) or may die near the end -- no checkpoint needs to exist yet. "
                         "The _cont<N>_SF suffix and --lora_checkpoint are derived at runtime.")
    ap.add_argument("--depends-on", default=None, dest="depends_on",
                    help="SLURM job id to queue behind; submits with --dependency=afterany:<id> "
                         "(afterany so it still runs if that job fails late). Implies the job waits.")
    ap.add_argument("--submit", action="store_true", help="sbatch the generated script")
    args = ap.parse_args()

    if not os.path.isfile(args.base):
        fail(f"base script not found: {args.base}")
    text = open(args.base).read()

    base_run = parse_arg_value(text, "output_path")
    if not base_run:
        fail("no --output_path found in base script")
    run_name = os.path.basename(base_run.rstrip("/"))           # container path /output/<run>
    container_out_root = os.path.dirname(base_run.rstrip("/"))  # e.g. /output

    # Cumulative step count from the very first run. A continuation run restarts its own
    # step counter at 0, so the latest checkpoint only reflects steps in *that* job. The
    # parent run name already encodes the total accumulated up to its start in `_cont<N>`
    # (0 if this is the first continuation); add the current checkpoint step to it.
    m_prev = re.search(r"_cont(\d+)_SF\d+$", run_name)
    prev_total = int(m_prev.group(1)) if m_prev else 0
    # Strip any prior _cont/_SF suffix so chained continuations don't grow unbounded.
    stripped = re.sub(r"_cont\d+_SF\d+$", "", run_name)

    if args.runtime_latest:
        # No on-disk resolution now: the emitted script picks the latest checkpoint
        # at launch. The checkpoint dir need not exist yet (source run may still be
        # training / queued behind it).
        new = apply_runtime_latest(text, run_name, stripped, prev_total,
                                   container_out_root, args.sf)
        resume_desc = "latest step-*.safetensors on disk at launch (runtime-resolved)"
        new_output_desc = f"{container_out_root}/{stripped}_cont<latest+{prev_total}>_SF{args.sf}"
        jobname = short_jobname(run_name, "latest", args.sf)
    else:
        host_outputs = resolve_host_outputs(text, args.outputs_dir)
        ckpt_dir = os.path.join(host_outputs, run_name)
        if args.step is not None:
            step = args.step
            ckpt = os.path.join(ckpt_dir, f"step-{step}.safetensors")
            if not os.path.isfile(ckpt):
                fail(f"requested checkpoint does not exist: {ckpt}")
        else:
            step, all_steps = latest_step(ckpt_dir)
            print(f"checkpoints in {run_name}: {', '.join(f'step-{s}' for s in all_steps)}")
            print(f"-> resuming from latest: step-{step}")
        total_step = prev_total + step
        new_run = f"{stripped}_cont{total_step}_SF{args.sf}"
        new_output_path = f"{container_out_root}/{new_run}"
        ckpt_container = f"{container_out_root}/{run_name}/step-{step}.safetensors"

        new = text
        # 1) job-name
        new = re.sub(r"(#SBATCH --job-name=)\S+(.*)",
                     lambda m: f"{m.group(1)}{short_jobname(run_name, total_step, args.sf)}"
                               f"            # continue {run_name} @ step-{step}, skip_frames {args.sf}",
                     new, count=1)
        # 2) output_path
        new = re.sub(r"(--output_path\s+)\S+", rf"\g<1>{new_output_path}", new, count=1)
        # 3) drop any existing checkpoint/skip_frames lines so we don't duplicate
        new = re.sub(r"\n\s*--lora_checkpoint\s+\S+\s*\\", "", new)
        new = re.sub(r"\n\s*--skip_frames\s+\S+\s*\\", "", new)
        # 4) insert fresh checkpoint + skip_frames right after --lora_rank <n> \
        m = re.search(r"(\n)(\s*)--lora_rank\s+\S+\s*\\", new)
        if not m:
            fail("could not find a `--lora_rank` line to anchor the inserted args")
        indent = m.group(2)
        insert = (f"{m.group(0)}"
                  f"\n{indent}--lora_checkpoint {ckpt_container} \\"
                  f"\n{indent}--skip_frames {args.sf} \\")
        new = new[:m.start()] + insert + new[m.end():]
        resume_desc = ckpt_container
        new_output_desc = new_output_path
        jobname = short_jobname(run_name, total_step, args.sf)
        if prev_total:
            print(f"  cumulative : {prev_total} (parent) + {step} (this checkpoint) = {total_step} total steps")

    out_path = args.out or next_script_path(args.base)
    with open(out_path, "w") as f:
        f.write(new)
    os.chmod(out_path, 0o755)

    print(f"\nwrote: {out_path}")
    print(f"  job-name   : {jobname}")
    print(f"  resume from: {resume_desc}")
    print(f"  new run    : {new_output_desc}")
    print(f"  skip_frames: {args.sf}")

    sbatch_cmd = ["sbatch"]
    if args.depends_on:
        sbatch_cmd.append(f"--dependency=afterany:{args.depends_on}")
    sbatch_cmd.append(out_path)

    if args.submit:
        r = subprocess.run(sbatch_cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(r.returncode)
    else:
        print(f"\nreview, then submit with:\n  {' '.join(sbatch_cmd)}")


if __name__ == "__main__":
    main()
