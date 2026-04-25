#!/bin/bash
#SBATCH --job-name=epg_sh_rf_lblcomb_shortfix_opt
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan_labels_combined_shortfix/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan_labels_combined_shortfix/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

SHARPSHOOTER_DIR=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter
mkdir -p "$SHARPSHOOTER_DIR/logs/rf_samchan_labels_combined_shortfix"

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd "$SHARPSHOOTER_DIR"

uv run --extra cu129 model_evaluation.py \
  --data_path /data/labs/hopelab/epg/epg_data/sharpshooter_parquet \
  --save_path "$SHARPSHOOTER_DIR/sharpshooter_rf_results" \
  --model_path rf/rf_samchan_labels_combined.py \
  --model_name rf_samchan_labels_combined_shortfix \
  --coarse_first_letter_labels \
  --optuna

echo "Job finished at: $(date)"
