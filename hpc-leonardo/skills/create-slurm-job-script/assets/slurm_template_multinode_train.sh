#!/bin/bash
# =============================================================================
# Multi-node training SLURM template (CONDA, no container) — LEONARDO Booster
# =============================================================================
# Usage: replace every `TODO_*` token, then sbatch.
# Verify with `grep -n 'TODO_' <script>` — must print nothing before submit.
#
# Use this template by default. Use the SIF variant only when the project
# requires a pre-built container image:
#   `slurm_template_multinode_train_sif.sh` (same dir)
#
# Background facts (rules / why each directive exists):
#   references/slurm.md          — QoS, accounts, partition, CPUs
#   references/hardware.md       — A100 sm_80, IB fabric, NCCL settings
#   references/storage-layout.md — $WORK / $SCRATCH layout
# =============================================================================

# --- SBATCH directives — every TODO_* MUST be replaced before sbatch -------
#SBATCH --job-name=TODO_jobname                # e.g. s2v_lora_4n
#SBATCH --output=/leonardo_scratch/large/userexternal/%u/slurm_logs/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/%u/slurm_logs/%x_%j.err
#SBATCH --account=TODO_account                 # AIFAC_F02_378 (case-insensitive)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=TODO_qos                         # see SKILL.md § QoS Selection for the 4 options + auto-pick rule
#SBATCH --nodes=TODO_nodes                     # 1, 2, 4, 8, 16, ...
#SBATCH --ntasks-per-node=1                    # accelerate spawns 4 subprocs per node
#SBATCH --cpus-per-task=32                     # Leonardo Booster: 32 CPU / 4 GPU per node
#SBATCH --gres=gpu:4                           # whole-node A100-SXM-64GB
#SBATCH --time=TODO_time                       # must respect QoS cap (see SKILL.md § QoS Selection)
#SBATCH --switches=TODO_switches               # see SKILL.md § QoS Selection (dbg → 2@00:10:00 | normal/lprod → 2@01:00:00)
#SBATCH --mail-type=END,FAIL                   # Leonardo SMTP works — END + FAIL only (START is too noisy)
#SBATCH --mail-user=alex.liu@valka.ai          # change for other users; SKILL.md owner default

set -eo pipefail

# --- srun CPU propagation (REQUIRED since SLURM 22.05+) --------------------
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# --- Paths ------------------------------------------------------------------
# Convention: code on $WORK (small, backed up), everything else on $SCRATCH
# under the same project name. work-side `runs/` and `outputs/` should be
# symlinks pointing at the scratch dirs below (one-time setup in SKILL.md).
PROJECT_DIR=TODO_project_dir                   # code: /leonardo_work/AIFAC_F02_378/$USER/<project>
SCRATCH_PROJECT_DIR=TODO_scratch_project_dir   # data/logs: /leonardo_scratch/large/userexternal/$USER/<project>
DATASET_BASE=TODO_dataset_base                 # dataset root (usually $SCRATCH/...; can be team-shared)
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_PROJECT_DIR/outputs/job_${SLURM_JOB_ID}}  # override via env for custom runs
DS_CONFIG=TODO_ds_config                       # ds_config*.json (or "" if not DeepSpeed)

# Defensive mkdir — slurm_logs (user-global at $SCRATCH/slurm_logs) MUST already
# exist (SBATCH directives open .out/.err before this point; one-time
# `mkdir -p /leonardo_scratch/large/userexternal/$USER/slurm_logs` is in
# SKILL.md Step 6). outputs/ and runs/ live per-project under scratch.
mkdir -p "$OUTPUT_DIR" "$SCRATCH_PROJECT_DIR/runs"
: "${TMPDIR:?TMPDIR is not set — SLURM should have set this; refusing to continue.}"

# --- Conda env --------------------------------------------------------------
CONDA_ROOT=/leonardo_work/AIFAC_F02_378/${USER}/conda/miniforge3
CONDA_ENV=TODO_conda_env                       # full path (e.g. .../conda/envs/diffsynth2) or just env name
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- Leonardo host CUDA toolkit (ONLY for DeepSpeed's nvcc version check) ---
# The conda env's torch wheel (cu128) bundles its own CUDA runtime. We need
# host CUDA_HOME + nvcc on PATH so DeepSpeed's installed_cuda_version() works.
# DO NOT prepend $CUDA_HOME/lib64 to LD_LIBRARY_PATH — that dir ships
# libnvidia-ml.so 560.35 which mismatches Leonardo's host driver and breaks
# nvidia-smi with "Failed to initialize NVML: Driver/library version mismatch".
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.6/none
export PATH=$CUDA_HOME/bin:$PATH

# --- NCCL — Leonardo Booster fabric (BSC ticket #428699 + Leonardo tuning) -
# All 4 HCAs are pure IB (not RoCE). NVLS off (A100 has no NVLink-Sharp).
# PXN disabled so all 4 HCAs go in parallel (12 → 60 GB/s aggregate).
export NCCL_SOCKET_IFNAME=ib                   # prefix match — covers ib0..ib3
export NCCL_IB_HCA=mlx5                        # prefix match — covers mlx5_0..mlx5_5
export NCCL_IB_DISABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=10
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_GID_INDEX=0                     # pure IB on Leonardo; use 3 only on RoCE clusters
export NCCL_PXN_DISABLE=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_MIN_NCHANNELS=4
export NCCL_DEBUG=WARN
export NCCL_RAS_ENABLE=0

# --- NCCL / PyTorch distributed debug knobs — UNCOMMENT when debugging -----
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,NET            # also: COLL,NET,ENV,ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export CUDA_LAUNCH_BLOCKING=1                # very slow; one debug iteration only

# --- Compiler + CUDA --------------------------------------------------------
export CC=gcc
export CXX=g++
export TORCH_CUDA_ARCH_LIST="8.0"              # A100 sm_80
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Offline + Python hygiene -----------------------------------------------
export HF_HUB_OFFLINE=1                        # compute nodes have no internet
# export DIFFSYNTH_SKIP_DOWNLOAD=true          # uncomment for DiffSynth-based projects
# export WANDB_DISABLED=true                   # uncomment to disable wandb attempts
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1                      # block ~/.local from leaking into sys.path

# --- Cache redirects → per-job $TMPDIR (RAM tmpfs, 10 GB, auto-cleaned) ----
export HF_HOME=$TMPDIR
export HF_HUB_CACHE=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR
export PYTHONPYCACHEPREFIX=$TMPDIR
export TORCH_EXTENSIONS_DIR=$TMPDIR
export CUDA_CACHE_PATH=$TMPDIR
export MPLCONFIGDIR=$TMPDIR
export TORCH_HOME=$TMPDIR
export XDG_CACHE_HOME=$TMPDIR
# NOTE: MODELSCOPE_CACHE intentionally NOT here — set below to $WAN_MODELS
# (pre-staged Wan checkpoints, multi-GB). Comment out for non-Wan projects.

# --- Wan project paths (comment out for non-Wan projects) -------------------
WAN_MODELS=/leonardo_work/AIFAC_F02_378/shared/wan_models
export WAN_MODELS
export MODELSCOPE_CACHE=$WAN_MODELS
export DIFFSYNTH_MODEL_BASE_PATH=$WAN_MODELS

# --- DeepSpeed (comment out + drop --use_deepspeed below if not used) ------
export ACCELERATE_USE_DEEPSPEED=true
export ACCELERATE_DEEPSPEED_CONFIG_FILE=$DS_CONFIG

# --- Distributed master discovery (Leonardo: hostname is fine) -------------
# NCCL bootstrap rides the management network briefly (~KBs of KV exchange);
# data flows over IB because NCCL_IB_HCA / NCCL_SOCKET_IFNAME are set.
# See LEONARDO-onboarding/docs/slurm-env-reference.md.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
TOTAL_PROCESSES=$(($SLURM_NNODES * $GPUS_PER_NODE))

echo "======================================"
echo "Job ID:          $SLURM_JOB_ID"
echo "Conda env:       $CONDA_ENV"
echo "Nodes:           $SLURM_NODELIST"
echo "Master:          $MASTER_ADDR:$MASTER_PORT (ib0)"
echo "Total Processes: $TOTAL_PROCESSES ($SLURM_NNODES nodes x $GPUS_PER_NODE GPUs)"
echo "Dataset:         $DATASET_BASE"
echo "Output:          $OUTPUT_DIR"
echo "TMPDIR:          $TMPDIR"
echo "======================================"
nvidia-smi -L
echo "[python]   $(command -v python)"
echo "[nvcc]     $(nvcc --version 2>&1 | tail -1)"
python -c "import torch; print(f'[torch] {torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}')"
echo "======================================"

cd "$PROJECT_DIR"

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
        # No defaults shipped — every project's flags are different.
"
