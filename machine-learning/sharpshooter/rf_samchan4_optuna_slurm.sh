#!/bin/bash
#SBATCH --job-name=epg_sharpshooter_rf_samchan4_optuna
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan4/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan4/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

SHARPSHOOTER_DIR=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter
mkdir -p "$SHARPSHOOTER_DIR/logs/rf_samchan4"

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd "$SHARPSHOOTER_DIR"

# Optuna rf_samchan4 with expanded ranges (chunk_seconds 1-25, num_freqs 1-25, num_estimators up to 512, max_features 0.2-1, overlap 0.25-0.85, max_lag 3-30)
uv run --extra cu129 model_evaluation.py \
  --data_path /data/labs/hopelab/epg/epg_data/sharpshooter_parquet \
  --save_path "$SHARPSHOOTER_DIR/sharpshooter_rf_results" \
  --model_path rf/rf_samchan.py \
  --model_name rf_samchan4 \
  --optuna

echo "Job finished at: $(date)"
