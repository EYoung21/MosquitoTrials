#!/bin/bash
#SBATCH --job-name=epg_job
#SBATCH --array=0-4                      # 5 tasks: fold 0,1,2,3,4
#SBATCH --output=/data/labs/hopelab/epg/logs/%x_%A_%a.out
#SBATCH --error=/data/labs/hopelab/epg/logs/%x_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu06

set -euo pipefail

# IMPORTANT: Slurm opens output/error files before the script runs.
# Make sure this exists BEFORE submitting:
#   mkdir -p /data/labs/hopelab/epg/logs

# Make fold-specific save path to avoid collisions between array tasks
SAVE_BASE="/home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/results"
DATA_PATH="/data/labs/hopelab/epg/tarsalis_data_clean"
CUDA_VERSION="cu129"
MODEL_PATH="${1}"
FOLD="${SLURM_ARRAY_TASK_ID}"
FLAGS="${@:2}"

# Print info for debugging
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM array task ID (fold): ${SLURM_ARRAY_TASK_ID:-unset}"

# Optional: speed up uv on shared filesystems
export UV_CACHE_DIR="/data/labs/hopelab/uv_cache"
mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  # NOTE: Requires outbound internet; if your cluster blocks it,
  # ask your admin to provide a uv module or preinstall it in your image.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

echo "${MODEL_PATH} evaluation (fold ${FOLD})"
MODEL_DIR="${MODEL_PATH}${FLAGS// /_}"
MODEL_DIR="${MODEL_DIR//--/_}"

echo "Flags: ${FLAGS}"
echo "Model dir: ${MODEL_DIR}"
echo "Save path: ${SAVE_BASE}/${MODEL_DIR}/fold_${FOLD}"
echo "Data path: ${DATA_PATH}"
echo "Running command: "
echo "uv run --extra ${CUDA_VERSION} model_eval.py --data_path ${DATA_PATH} \
    --save_path ${SAVE_BASE}/${MODEL_DIR}/fold_${FOLD} --model_path ${MODEL_PATH}.py --model_name=${MODEL_PATH} ${FLAGS}  --optuna --fold \"${FOLD}\""

mkdir -p ${SAVE_BASE}/${MODEL_DIR}
mkdir -p ${SAVE_BASE}/${MODEL_DIR}/fold_${FOLD}
uv run --extra ${CUDA_VERSION} model_eval.py --data_path ${DATA_PATH} \
    --save_path ${SAVE_BASE}/${MODEL_DIR}/fold_${FOLD} --model_path ${MODEL_PATH}.py --model_name=${MODEL_PATH} ${FLAGS}  --optuna --fold "${FOLD}"
uv run summarize_folds.py --save_path ${SAVE_BASE}/${MODEL_DIR} --model_name=${MODEL_PATH} ${FLAGS} --num_folds 5

echo "Job finished at: $(date)"
