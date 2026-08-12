#!/bin/bash
# =============================================================================
# Multi-node training SLURM template (SIF / Singularity) — LEONARDO Booster
# =============================================================================
# Usage: replace every `TODO_*` token, then sbatch.
# Verify with `grep -n 'TODO_' <script>` — must print nothing before submit.
#
# Use the conda variant (`slurm_template_multinode_train.sh`) by default.
# Use this SIF variant when you need a pinned container image (reproducibility,
# CUDA forward-compat baked in, JIT-free deepspeed builds inside the SIF, …).
#
# Default SIF (rebuild via the .def file alongside): the diffsynth-a100 SIF
# at /leonardo_work/AIFAC_F02_378/shared/singularity/diffsynth-a100/
#   diffsynth-bind-a100.sif. Its %environment bakes in NCCL settings,
#   CUDA forward-compat, and TMPDIR cache redirects — see the .def file.
#   This SLURM script only injects project-level vars via `--env`.
# =============================================================================

# --- SBATCH directives — every TODO_* MUST be replaced before sbatch -------
#SBATCH --job-name=TODO_jobname                # e.g. s2v_lora_8n
#SBATCH --output=/leonardo_scratch/large/userexternal/%u/slurm_logs/%x_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/%u/slurm_logs/%x_%j.err
#SBATCH --account=TODO_account                 # AIFAC_F02_378 (case-insensitive)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=TODO_qos                         # see SKILL.md § QoS Selection for the 4 options + auto-pick rule
#SBATCH --nodes=TODO_nodes                     # 1, 2, 4, 8, 16, ...
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32                     # Leonardo Booster: 32 CPU / 4 GPU per node
#SBATCH --gres=gpu:4                           # whole-node A100-SXM-64GB
#SBATCH --time=TODO_time                       # must respect QoS cap (see SKILL.md § QoS Selection)
#SBATCH --switches=TODO_switches               # see SKILL.md § QoS Selection (dbg → 2@00:10:00 | normal/lprod → 2@01:00:00)
#SBATCH --mail-type=END,FAIL                   # Leonardo SMTP works — END + FAIL only
#SBATCH --mail-user=alex.liu@valka.ai          # change for other users; SKILL.md owner default

set -eo pipefail

# --- srun CPU propagation ---------------------------------------------------
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# --- Paths ------------------------------------------------------------------
# Convention: code on $WORK (small, backed up), everything else on $SCRATCH
# under the same project name. work-side `runs/` and `outputs/` should be
# symlinks pointing at the scratch dirs below (one-time setup in SKILL.md).
PROJECT_DIR=TODO_project_dir                                                       # code: /leonardo_work/AIFAC_F02_378/$USER/<project> (bound → /workspace)
SCRATCH_PROJECT_DIR=TODO_scratch_project_dir                                    # data/logs: /leonardo_scratch/large/userexternal/$USER/<project>
SIF=/leonardo_work/AIFAC_F02_378/shared/singularity/diffsynth-a100/diffsynth-bind-a100.sif
MODELS_DIR=/leonardo_work/AIFAC_F02_378/shared/wan_models                       # WAN_MODELS / MODELSCOPE_CACHE source
DATA_BASE=TODO_dataset_base                                                     # dataset root (usually $SCRATCH/...; can be team-shared)
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH_PROJECT_DIR/outputs/job_${SLURM_JOB_ID}}      # override via env for custom runs

# DS_CONFIG path is from the CONTAINER's view because torchrun runs inside SIF.
# With $PROJECT_DIR bound to /workspace, point to /workspace/<rel-path>:
#   e.g. /workspace/livedealer/train/lora/ds_config_zero2_8node.json
DS_CONFIG_IN_CONTAINER=TODO_ds_config_in_container

# Defensive mkdir — slurm_logs (user-global at $SCRATCH/slurm_logs) MUST already
# exist before sbatch (one-time `mkdir -p /leonardo_scratch/large/userexternal/
# $USER/slurm_logs` is in SKILL.md Step 6). outputs/ and runs/ live per-project.
mkdir -p "$OUTPUT_DIR" "$SCRATCH_PROJECT_DIR/runs"
: "${TMPDIR:?TMPDIR is not set — SLURM should have set this; refusing to continue.}"
[ -r "$SIF" ] || { echo "ERROR: SIF not readable: $SIF" >&2; exit 1; }

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
echo "Container:       $SIF"
echo "Nodes:           $SLURM_NODELIST"
echo "Master:          $MASTER_ADDR:$MASTER_PORT (ib0)"
echo "Total Processes: $TOTAL_PROCESSES ($SLURM_NNODES nodes x $GPUS_PER_NODE GPUs)"
echo "Dataset:         $DATA_BASE"
echo "Output:          $OUTPUT_DIR"
echo "TMPDIR (host):   $TMPDIR"
echo "======================================"
nvidia-smi -L
echo "======================================"

cd "$PROJECT_DIR"

# =============================================================================
# Bind mounts (host → container):
#   $PROJECT_DIR        → /workspace
#   $MODELS_DIR      → /wan_models    (WAN_MODELS / MODELSCOPE_CACHE point here)
#   $DATA_BASE       → /data (ro)
#   $OUTPUT_DIR      → /output
#   $TMPDIR          → /tmp           (SIF %environment routes HF/Triton caches here)
#
# Vars passed via --env (SIF def deliberately leaves these to the SLURM script
# per-project; everything cluster-generic — NCCL, CUDA forward-compat, etc. —
# is already baked into the SIF %environment block):
#   WAN_MODELS / MODELSCOPE_CACHE  — point to /wan_models (the bound dir)
#   NCCL_DEBUG                      — override for visibility (default WARN)
# =============================================================================
srun --kill-on-bad-exit=1 \
  singularity exec --nv \
    --bind "${PROJECT_DIR}:/workspace" \
    --bind "${MODELS_DIR}:/wan_models" \
    --bind "${DATA_BASE}:/data:ro" \
    --bind "${OUTPUT_DIR}:/output" \
    --bind "${TMPDIR}:/tmp" \
    --env WAN_MODELS=/wan_models \
    --env MODELSCOPE_CACHE=/wan_models \
    --env NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    "$SIF" \
    /bin/bash -c '
        set -eo pipefail
        cd /workspace

        export DIFFSYNTH_MODEL_BASE_PATH=/wan_models
        export DIFFSYNTH_SKIP_DOWNLOAD=true
        export ACCELERATE_USE_DEEPSPEED=true
        export ACCELERATE_DEEPSPEED_CONFIG_FILE='"$DS_CONFIG_IN_CONTAINER"'

        echo "[python] $(command -v python)"
        echo "[nvcc]   $(nvcc --version 2>&1 | tail -1)"
        # CUDA forward-compat visibility (SIF bundles libcuda.so.570 from
        # /usr/local/cuda/compat/, shadowing the older host driver on
        # Leonardo). These 3 lines make it obvious from the log whether
        # the bundled libcuda or the host one wins the dlopen race.
        echo "[host-driver]    $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo unknown)"
        _compat=$(ls -1 /usr/local/cuda/compat/libcuda.so.* 2>/dev/null | head -1)
        echo "[libcuda-compat] ${_compat##*/}"
        echo "[LD_LIB head 2]: $(echo $LD_LIBRARY_PATH | cut -d: -f1-2)"
        python -c "import torch; print(f\"[torch] {torch.__version__} cuda={torch.cuda.is_available()} n_gpu={torch.cuda.device_count()}\")"

        accelerate launch \
            --use_deepspeed \
            --deepspeed_multinode_launcher standard \
            --deepspeed_config_file '"$DS_CONFIG_IN_CONTAINER"' \
            --num_processes '"$TOTAL_PROCESSES"' \
            --num_machines '"$SLURM_NNODES"' \
            --machine_rank $SLURM_NODEID \
            --main_process_ip '"$MASTER_ADDR"' \
            --main_process_port '"$MASTER_PORT"' \
            --mixed_precision bf16 \
            TODO_path_in_container/train_script.py \
            TODO_train_script_flags
            # ↑ TODO_path_in_container should start with /workspace/...
            # ↑ TODO_train_script_flags = project-specific CLI flags; no defaults shipped.
    '
