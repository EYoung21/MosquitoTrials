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

set -euo pipefail

# IMPORTANT: Slurm opens output/error files before the script runs.
# Make sure this exists BEFORE submitting:
#   mkdir -p /data/labs/hopelab/epg/logs

FOLD="${SLURM_ARRAY_TASK_ID}"

# Print info for debugging
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID:-unset}"
echo "SLURM array task ID (fold): ${SLURM_ARRAY_TASK_ID:-unset}"

# Optional: speed up uv on shared filesystems
export UV_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/${USER}/uv-cache"
mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  # NOTE: Requires outbound internet; if your cluster blocks it,
  # ask your admin to provide a uv module or preinstall it in your image.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

echo "UNET attention evaluation (fold ${FOLD})"

# Make fold-specific save path to avoid collisions between array tasks
SAVE_BASE="/home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_attention"
SAVE_PATH="${SAVE_BASE}/fold_${FOLD}"

# uv run --extra cu129 model_eval.py \
#   --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#   --save_path "$SAVE_PATH" \
#   --model_path unet.py \
#   --model_name unet \
#   --attention \
#   --optuna \
#   --fold "${FOLD}"


echo "UNET CRF evaluation"
mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_crf
mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_crf/fold_${FOLD}
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_crf/fold_${FOLD} --model_path unet_crf.py --model_name=unet_crf  --optuna --fold "${FOLD}"

echo "UNET CRF attention evaluation"
mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_attention_crf
mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_attention_crf/fold_${FOLD}
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_attention_crf/fold_${FOLD} --model_path unet_crf.py --model_name=unet_crf --attention  --optuna --fold "${FOLD}"

# echo "Random forest evaluation"
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_forest
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_forest/fold_${FOLD}
# uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#     --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_forest/fold_${FOLD} --model_path rf.py --model_name=rf --optuna --fold "${FOLD}" # --post_process v
# # # --optuna

# echo "UNET evaluation"
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet/fold_${FOLD}
# uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#     --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet/fold_${FOLD} --model_path unet.py --model_name=unet  --optuna --fold "${FOLD}"

# echo "TCN evaluation"
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_tcn
# mkdir -p  /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_tcn/fold_${FOLD}
# uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#     --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_tcn/fold_${FOLD} --model_path tcn.py --model_name=tcn  --optuna --fold "${FOLD}"# --post_process v

# echo "Transformer evaluation"
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_transformer
# mkdir -p /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_transformer/fold_${FOLD}
# uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#     --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_transformer/fold_${FOLD} --model_path transformer.py --model_name=transformer  --optuna --fold "${FOLD}"# --post_process v



echo "Job finished at: $(date)"
