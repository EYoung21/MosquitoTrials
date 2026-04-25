#!/bin/bash
#SBATCH --job-name=epg_rf_smote_lb05_optuna
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

MOSQUITO_DIR=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito
mkdir -p "$MOSQUITO_DIR/eliIntroLabsResults/logs"

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd "$MOSQUITO_DIR"

uv run --extra cu129 model_eval.py \
  --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
  --save_path "$MOSQUITO_DIR/eliIntroLabsResults/tarsalis_results_smote_lb05" \
  --model_path rf_sunwindow_smote_lb05.py \
  --model_name rf_sunwindow_smote_lb05 \
  --optuna \
  --augment

echo "Job finished at: $(date)"
