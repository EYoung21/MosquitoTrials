#!/bin/bash
#SBATCH --job-name=epg_sh_rf_samchan_lbl_dir
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan_labels_combined_directional/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/logs/rf_samchan_labels_combined_directional/%x_%j.err
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
mkdir -p "$SHARPSHOOTER_DIR/logs/rf_samchan_labels_combined_directional"

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd "$SHARPSHOOTER_DIR"

# First-letter label collapse (F1,F2->F, etc.): model + DataImport must agree
# Directional boundary expansion in rf_samchan_labels_combined.py:
# - chunk_seconds max raised beyond prior best-at-max
# - num_estimators extended above 1024
# - max_features min lowered below 0.2
uv run --extra cu129 model_evaluation.py \
  --data_path /data/labs/hopelab/epg/epg_data/sharpshooter_parquet \
  --save_path "$SHARPSHOOTER_DIR/sharpshooter_rf_results" \
  --model_path rf/rf_samchan_labels_combined.py \
  --model_name rf_samchan_labels_combined_directional \
  --coarse_first_letter_labels \
  --optuna

echo "Job finished at: $(date)"
