#!/bin/bash
# =============================================================================
# Multi-node training SLURM template — BSC MareNostrum5, acc partition
# =============================================================================
# Usage: replace every `TODO_*` token, then sbatch.
# Verify with `grep -n 'TODO_' <script>` — must print nothing before submit.
#
# Background facts (rules / why each directive exists):
#   docs/slurm.md              — QoS, accounts, switches, CPUs
#   docs/hardware.md           — H100, NVLink, NCCL fabric, NUMA
#   docs/gpfs-resilience.md    — D-state probe + page-cache warmup
#   docs/job-notifier.md       — Slack EXIT trap
# =============================================================================

# --- SBATCH directives — every TODO_* MUST be replaced before sbatch -------
#SBATCH --job-name=TODO_jobname                      # e.g. wan22vae_stage1_4n
#SBATCH --output=/gpfs/scratch/ehpc1003/%u/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/ehpc1003/%u/slurm_logs/%x_%j.err
#SBATCH --nodes=TODO_nodes                           # 1, 2, 4, 8, 16, 32, 64
#SBATCH --ntasks-per-node=1                          # accelerate spawns 4 subprocs
#SBATCH --cpus-per-task=80                           # BSC: 4 GPUs × 20 CPUs/GPU
#SBATCH --gres=gpu:4                                 # whole node only
#SBATCH --account=TODO_account                       # from `bsc_acct`; rotates per project
#SBATCH --qos=TODO_qos                               # acc_debug | acc_ehpc
#SBATCH --switches=TODO_switches                     # acc_debug: 2@0:10:00 | acc_ehpc: 2@1:00:00
#SBATCH --time=TODO_time                             # acc_debug ≤02:00:00 | acc_ehpc ≤3-00:00:00

# -----------------------------------------------------------------------------
# Module loads (BSC ACC standard set)
# -----------------------------------------------------------------------------
module purge
module load bsc/1.0
module load miniforge/25.3.0-3
module load nvidia-hpc-sdk/23.9
module load gcc/11.4.0
module load cuda/12.8
module load nccl/2.20.5                              # kept as defensive fallback (see slurm.md § 9)
module load binutils/2.37
module load ucx/1.19.0

# --- Slack EXIT notifier (acc_ehpc only, fail-silent) -----------------------
# See docs/job-notifier.md. The notifier reads ~/.slack_webhook (chmod 600) and
# fires on normal exit, exit N, srun fail, SIGTERM (walltime/scancel) — but not
# SIGKILL or node crash. Safe no-op when ~/.slack_webhook is missing.
source $HOME/scripts/job_notifier

# --- srun CPU propagation (REQUIRED since SLURM 22.05+; see slurm.md § 8) ---
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# --- NCCL on BSC ACC fabric (validated, see hardware.md § NCCL settings) ----
# These values come from BSC support ticket #428699 (Jon Navarro, 2026-04-16):
# "problem-free and good performance on the ACC fabric." Don't override unless
# a newer BSC-support recommendation arrives.
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0,ib1,ib2,ib3            # all 4 IPoIB interfaces (bootstrap)
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5       # 4 RDMA HCAs (data plane)
export NCCL_NVLS_ENABLE=0                            # disable NVLink Sharp (flaky on BSC)
export NCCL_NET_GDR_LEVEL=2                          # GPUDirect RDMA over RoCE
export NCCL_IB_GID_INDEX=3                           # required for BSC RoCEv2
export NCCL_RAS_ENABLE=0                             # disable RAS (NCCL 2.20+ stability)

# --- NCCL / PyTorch distributed debug knobs — UNCOMMENT when debugging -----
# Pick what you need, then remove the leading `#`. Lots of lines per rank.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET            # also: COLL,NET,ENV,ALL
# export NCCL_DEBUG_FILE=/gpfs/scratch/ehpc1003/${USER}/slurm_logs/nccl_${SLURM_JOB_ID}_%h_%p.log
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_LAUNCH_BLOCKING=1                # very slow; one debug iteration only
# export NCCL_IB_DISABLE=1                     # last-resort: TCP fallback (~10× slower)

# --- Compiler + CUDA --------------------------------------------------------
export CC=gcc                                  # DeepSpeed JIT prefers gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST="9.0a"             # H100 only (see hardware.md)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Offline / no-download switches (compute nodes have no internet) --------
export HF_HUB_OFFLINE=1                 # transformers ≥4.36 honors this; covers all hub downloads
# export DIFFSYNTH_SKIP_DOWNLOAD=true   # uncomment for DiffSynth-based projects
# export WANDB_DISABLED=true            # uncomment if you don't want wandb attempts

# --- Cache redirects → per-job $TMPDIR (auto-cleaned at job end) ------------
# Fail loudly if SLURM forgot to set $TMPDIR (would silently write to "").
: "${TMPDIR:?TMPDIR is not set — SLURM should have set this; refusing to continue.}"
export HF_HOME=$TMPDIR                  # umbrella; HF derives transformers/datasets/hub from this
export HF_HUB_CACHE=$TMPDIR             # explicit hub-download cache
# NOTE: HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE are deprecated;
# transformers ≥4.36 emits a FutureWarning if either is set.
export TRITON_CACHE_DIR=$TMPDIR

# Python bytecode (.pyc) → $TMPDIR instead of __pycache__/ next to source.
# Avoids GPFS metadata overhead from many tiny .pyc files in project dirs,
# avoids stray __pycache__/ accumulating in /gpfs/projects, and prevents
# write-permission errors when project trees are read-only.
export PYTHONPYCACHEPREFIX=$TMPDIR
export TORCH_EXTENSIONS_DIR=$TMPDIR     # DeepSpeed JIT lock contention
export CUDA_CACHE_PATH=$TMPDIR          # PTX→cubin race
export MPLCONFIGDIR=$TMPDIR             # matplotlib first-import race
export DG_CACHE_DIR=$TMPDIR             # DeepGEMM JIT
export FLASHINFER_JIT_DIR=$TMPDIR       # FlashInfer JIT
export VLLM_CACHE_ROOT=$TMPDIR          # vLLM compile cache
export TORCH_HOME=$TMPDIR               # torch.hub
export XDG_CACHE_HOME=$TMPDIR           # catch-all (pip, uv, wandb, ...)
# NOTE: MODELSCOPE_CACHE intentionally NOT here — set below to $WAN_MODELS
# (pre-staged Wan checkpoints, ~159 GB). Comment out for non-Wan projects.

# --- Distributed master discovery (ib0 IP, not hostname) -------------------
# torchrun / accelerate need an IP that's reachable over the IB fabric.
# Plain $(hostname) returns a name that resolves to a slow management network
# on BSC; ib0 is the correct interface for inter-node rendezvous.
MASTER_HOSTNAME=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_HOSTNAME" \
    bash -lc "ip -4 addr show ib0 | awk '/inet / {print \$2}' | cut -d/ -f1" | tail -n 1)
export MASTER_PORT=29500   # PyTorch/torchrun/accelerate convention

# Fail fast if MASTER_ADDR couldn't be resolved (ib0 missing / srun issue).
# Empty MASTER_ADDR would make torch.distributed hang for hours.
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: failed to resolve ib0 IP for MASTER_ADDR on $MASTER_HOSTNAME" >&2
    echo "       Check: ib0 interface, srun access, scontrol show hostnames output." >&2
    exit 1
fi

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
TOTAL_PROCESSES=$(($SLURM_NNODES * $GPUS_PER_NODE))

echo "======================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Nodes:           $SLURM_NODELIST"
echo "Master:          $MASTER_ADDR:$MASTER_PORT (ib0)"
echo "Total Processes: $TOTAL_PROCESSES ($SLURM_NNODES nodes x $GPUS_PER_NODE GPUs)"
echo "======================================"

# --- Project paths — every TODO_* MUST be replaced -------------------------
WAN_MODELS=/gpfs/scratch/ehpc1003/${USER}/wan_models                           # leave as-is for Wan projects; comment out for non-Wan
DATASET_BASE=TODO_dataset_base                                                # absolute path to dataset root
OUTPUT_DIR=TODO_output_dir                                                    # absolute path; mkdir -p done below
PROJECT_DIR=TODO_project_dir                                                  # absolute path to project (under ~/projects/)

mkdir -p "/gpfs/scratch/ehpc1003/${USER}/slurm_logs"
mkdir -p "$OUTPUT_DIR"

# modelscope reads pre-staged Wan checkpoints from $WAN_MODELS (org/repo/ layout)
export MODELSCOPE_CACHE=$WAN_MODELS

# --- Conda activation (once on BatchHost; --export=ALL propagates) ---------
source /gpfs/apps/MN5/ACC/MINIFORGE/25.3.0-3/etc/profile.d/conda.sh
conda activate TODO_conda_env                                                 # env name under ~/scratch/envs/
export PATH="$CONDA_PREFIX/bin:$PATH"
export IMAGEIO_FFMPEG_EXE=$CONDA_PREFIX/bin/ffmpeg                            # avoid imageio bundled binary
echo "[ffmpeg-check] which ffmpeg: $(which ffmpeg 2>/dev/null || echo NOT_FOUND)"

# --- DeepSpeed config — every TODO_* MUST be replaced ----------------------
DS_CONFIG=TODO_ds_config                                                       # absolute path to ds_config*.json; or comment line + drop --deepspeed_config_file below if no DeepSpeed

# --- Page-cache warmup (per-node, parallel) --------------------------------
# Fixes multi-node rendezvous TCPStore 900s timeout from cold-import GPFS storms.
# warmup_page_cache.sh: probes GPFS first (fail-fast on D-state nodes); cats
# ~5GB site-packages into kernel page cache. Empirically ~30-60s on a cold node.
# See docs/gpfs-resilience.md for the why.
echo "======================================"
echo "Page cache warmup (per node, parallel)"
echo "======================================"
WARMUP_T0=$SECONDS
srun --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash $HOME/scripts/warmup_page_cache.sh "$CONDA_PREFIX"
echo "[warmup] aggregate wall-clock: $((SECONDS - WARMUP_T0))s"

# =============================================================================
# Run training (multi-node srun + per-node accelerate launch)
# =============================================================================
# The accelerate flags below (--num_processes / --num_machines / etc.) are
# generic to all multi-node DDP / DeepSpeed launches. The script's own flags
# (--lr, --batch-size, dataset paths, model config, etc.) are project-specific
# and MUST be filled in by you for each project — this template doesn't ship
# any defaults for them.
# =============================================================================
srun --kill-on-bad-exit=1 bash -c "
    echo \"Node \${SLURM_NODEID}/\${SLURM_NNODES} (\$(hostname)) master=$MASTER_ADDR launching...\"

    cd $PROJECT_DIR
    accelerate launch \
        --use_deepspeed \
        --deepspeed_multinode_launcher standard \
        --deepspeed_config_file $DS_CONFIG \
        --num_processes $TOTAL_PROCESSES \
        --num_machines $SLURM_NNODES \
        --machine_rank \$SLURM_NODEID \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --mixed_precision bf16 \
        TODO_path/to/train_script.py \
        TODO_train_script_flags
        # ↑ replace TODO_path/... with your training script (relative to \$PROJECT_DIR)
        # ↑ replace TODO_train_script_flags with your script's CLI args, e.g.:
        #     --output_path \$OUTPUT_DIR \\
        #     --dataset_base_path \$DATASET_BASE \\
        #     --learning_rate 5e-5 \\
        #     --num_epochs 10 \\
        #     ...
        # No defaults shipped — every project's flags are different.
"
