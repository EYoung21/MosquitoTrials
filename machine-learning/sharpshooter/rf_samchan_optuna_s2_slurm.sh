#!/bin/bash
#SBATCH --job-name=epg_sharpshooter_rf_samchan_s2_optuna
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan_s2/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan_s2/%x_%j.err
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
mkdir -p "$SHARPSHOOTER_DIR/logs/rf_samchan_s2"

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd "$SHARPSHOOTER_DIR"

# s2: s1 historical bounds with expanded max_lag upper bound
uv run --extra cu129 model_evaluation.py \
  --data_path /data/labs/hopelab/epg/epg_data/sharpshooter_parquet \
  --save_path "$SHARPSHOOTER_DIR/sharpshooter_rf_results" \
  --model_path rf/rf_samchan_s2.py \
  --model_name rf_samchan_s2 \
  --optuna

echo "Job finished at: $(date)"

