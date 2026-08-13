#!/usr/bin/env python3
"""Generate (and optionally launch) a real-streaming S2V inference *performance
benchmark* — the streaming pipeline run with --profile + --gpu_telemetry so it
drops gpu_telemetry.csv (clock/power/temp) and module_fps.csv (per-module fps)
into the save dir for later analysis by summarize_bench.py.

This mirrors live_dealer/infer/livedealer_infer_real_streaming_lambda.sh (env
block + torchrun of livedealer_infer_real_streaming.py) but parametrizes the
perf knobs and FORCES --profile --gpu_telemetry. It is the streaming pipeline
ONLY — there is no --vae_only isolation mode here.

Run:
    python3 .claude/skills/benchmark-streaming/run_bench.py \
        [--steps 3] [--cuda-visible-devices 1] \
        [--width 1280 --height 720 --fps 24] \
        [--frames-per-clip 12 --lframes-per-block 3 --lframes-per-kv-cache 3] \
        [--num-clips 25 | --sustained] [--repeat 1] \
        [--lora-path PATH] [--input-image P --pose-video P --audio-path P --card-detection P] \
        [--vae-encoder-type wanvae2.1] [--mem-profile] [--save-path DIR] \
        [--launch]        # run it (background-friendly); tees to <save>/bench.log

Prints the save_path on the last stdout line as `SAVE_PATH=<dir>` so the caller
can hand it straight to summarize_bench.py.
"""
import argparse
import os
import shlex
import subprocess
import sys

# Defaults mirror the perf case in livedealer_infer_real_streaming_lambda.sh
# (talking/dealer mode-switch example: eyes-only pose + its audio + card list).
D_LORA = "models/train/leo_card_class_embed_16nodes_nomask_fixalign_fresh_cont7920_SF150_pfm_iv2/step-5600.safetensors"
D_IMAGE = "data/live_dealer/test/live_dealer_5_1_2026_1.jpg"
D_MASK = "data/live_dealer/test/ref_msk/mask.png"
D_POSE = "data/live_dealer/test/examples/infer_list_shape_body_only_eye_only.txt"
D_AUDIO = "data/live_dealer/test/examples/infer_list_audio.txt"
D_CARD = "data/live_dealer/test/examples/infer_list_card_detection.txt"

ENTRY = "live_dealer/infer/livedealer_infer_real_streaming.py"


def build(a):
    ndev = len([x for x in a.cuda_visible_devices.split(",") if x != ""])
    name = (f"{a.lframes_per_block}bf{a.lframes_per_kv_cache}kvf_"
            f"{a.width}-{a.height}_{a.fps}fps_{a.steps}step_{ndev}gpu")
    if a.mem_profile:
        name += "_mem"

    if a.save_path:
        save_path = a.save_path
    else:
        run = os.path.splitext(os.path.basename(a.lora_path))[0] if a.lora_path else "no_lora"
        lora_run = os.path.basename(os.path.dirname(a.lora_path)) if a.lora_path else "no_lora"
        save_path = f"output/bench/{lora_run}/{run}/{name}"

    num_clips = a.num_clips
    repeat = a.repeat
    if a.sustained:
        num_clips = 1000
        repeat = 0  # loop forever — for thermal/steady-state curves, stop with Ctrl-C

    def opt(flag, val):
        return [flag, val] if val not in (None, "", "None") else []

    torchrun = ["torchrun", "--standalone", f"--nproc_per_node={ndev if ndev else 'gpu'}",
                "--master_addr=127.0.0.1", f"--master_port={a.master_port}", ENTRY,
                "--input_image", a.input_image]
    torchrun += opt("--input_mask", a.input_mask)
    torchrun += opt("--lora_path", a.lora_path)
    torchrun += opt("--pose_video", a.pose_video)
    torchrun += opt("--audio_path", a.audio_path)
    torchrun += opt("--card_detection", a.card_detection)
    torchrun += [
        "--save_path", save_path,
        "--height", str(a.height), "--width", str(a.width), "--fps", str(a.fps),
        "--num_clips", str(num_clips),
        "--frames_per_clip", str(a.frames_per_clip),
        "--lframes_per_block", str(a.lframes_per_block),
        "--lframes_per_kv_cache", str(a.lframes_per_kv_cache),
        "--num_inference_steps", str(a.steps),
        "--repeat", str(repeat),
        "--vae_encoder_type", a.vae_encoder_type,
        "--cfg_scale", "1.0",
        "--no_warmup_stream" if a.no_warmup else None,
        # ↓↓↓ the benchmark's whole point — always on ↓↓↓
        "--profile", "--gpu_telemetry",
        "--gpu_telemetry_interval", str(a.telemetry_interval),
    ]
    if a.mem_profile:
        torchrun.append("--mem_profile")
    torchrun = [t for t in torchrun if t is not None]

    env = [
        "source .venv/bin/activate",
        'mkdir -p "${HOME}/.triton/autotune"',
        'export TRITON_CACHE_DIR="${HOME}/.triton"',
        "export HF_HUB_OFFLINE=1",
        "export PYTHONUNBUFFERED=1",
        "export TOKENIZERS_PARALLELISM=false",
        f"export CUDA_VISIBLE_DEVICES={a.cuda_visible_devices}",
        "export CUDA_DEVICE_MAX_CONNECTIONS=1",
        "export OMP_NUM_THREADS=8",
        "export ENABLE_COMPILE=True",
        "export COUNT_COMPILE_GRAPHS=0",
        "export WAN_FAST_VAE=True",
        "export NCCL_DEBUG=WARN",
        "export NCCL_P2P_LEVEL=NVL",
        "export NCCL_TIMEOUT=7200",
        "export NCCL_ASYNC_ERROR_HANDLING=1",
        "export MASTER_ADDR=127.0.0.1",
        f"export MASTER_PORT={a.master_port}",
    ]
    snippet = "\n".join(env) + "\n\n" + " \\\n    ".join(torchrun) + "\n"
    return snippet, save_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3, help="num_inference_steps (with GPU count picks the partition preset)")
    p.add_argument("--cuda-visible-devices", default="1", help="CUDA_VISIBLE_DEVICES (count = nproc_per_node)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frames-per-clip", type=int, default=12)
    p.add_argument("--lframes-per-block", type=int, default=3)
    p.add_argument("--lframes-per-kv-cache", type=int, default=3)
    p.add_argument("--num-clips", type=int, default=25, help="finite bounded run (~10s @24fps for 25 clips)")
    p.add_argument("--sustained", action="store_true", help="loop forever (num_clips=1000, repeat=0) for thermal curves; Ctrl-C to stop")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--lora-path", default=D_LORA)
    p.add_argument("--input-image", default=D_IMAGE)
    p.add_argument("--input-mask", default=D_MASK)
    p.add_argument("--pose-video", default=D_POSE)
    p.add_argument("--audio-path", default=D_AUDIO)
    p.add_argument("--card-detection", default=D_CARD)
    p.add_argument("--vae-encoder-type", default="wanvae2.1")
    p.add_argument("--mem-profile", action="store_true", help="also emit --mem_profile (per-rank memory breakdown → memory_report.txt)")
    p.add_argument("--no-warmup", action="store_true", help="emit --no_warmup_stream (first clip absorbs the compile cost)")
    p.add_argument("--telemetry-interval", type=float, default=1.0)
    p.add_argument("--master-port", type=int, default=29501)
    p.add_argument("--save-path", default=None, help="override the auto-derived output/bench/... dir")
    p.add_argument("--launch", action="store_true", help="run the benchmark now (tees to <save>/bench.log); best invoked as a background Bash command")
    a = p.parse_args()

    snippet, save_path = build(a)

    print("# ---- streaming benchmark ----", file=sys.stderr)
    print(f"# save_path: {save_path}", file=sys.stderr)
    print(snippet)

    if a.launch:
        os.makedirs(save_path, exist_ok=True)
        log = os.path.join(save_path, "bench.log")
        # Run through bash so `source`, exports and the line-continuations all apply.
        cmd = f"set -euo pipefail\n{snippet}\n"
        print(f"# launching → {log}", file=sys.stderr)
        with open(log, "w") as lf:
            proc = subprocess.run(["bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT)
        print(f"# exit={proc.returncode} | log={log}", file=sys.stderr)
        print(f"SAVE_PATH={save_path}")
        sys.exit(proc.returncode)

    print(f"SAVE_PATH={save_path}")


if __name__ == "__main__":
    main()
