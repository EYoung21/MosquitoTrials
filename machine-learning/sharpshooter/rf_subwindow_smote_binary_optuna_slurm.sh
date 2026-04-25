#!/bin/bash
#SBATCH --job-name=epg_sharpshooter_rf_binary_optuna
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_binary_optuna/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_binary_optuna/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1

# Optuna for binary P (probing) vs NP (non-probing) — ML model for probe splitter.
# Uses preprocessed sharpshooter parquet; full recordings (no probe segmentation) so both classes are present.

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

SHARPSHOOTER_DIR=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter
mkdir -p "$SHARPSHOOTER_DIR/logs/rf_binary_optuna"

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd "$SHARPSHOOTER_DIR"

uv run --extra cu129 model_evaluation.py \
  --data_path /data/labs/hopelab/epg/epg_data/sharpshooter_parquet \
  --save_path "$SHARPSHOOTER_DIR/sharpshooter_rf_binary_results" \
  --model_path rf/rf_subwindow_smote.py \
  --model_name rf_subwindow_smote_binary \
  --binary \
  --optuna

echo "Job finished at: $(date)"
