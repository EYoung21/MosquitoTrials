#!/bin/bash
#SBATCH --job-name=epg_rf_smote_edge_dir
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

# Directional boundary expansion (toward where the best trial sat):
# - augment_factor includes lower-than-1 options (0.5, 0.75) in model_eval.py
# - chunk_seconds minimum reduced (now 0.1 in rf_sunwindow_smote.py)
# - smote_k_neighbors maximum increased (now 50 in rf_sunwindow_smote.py)

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

mkdir -p /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs

echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

cd /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito

uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/tarsalis_results_smote_edge_directional \
    --model_path rf_sunwindow_smote.py --model_name=rf_sunwindow_smote_edge_directional --optuna --augment

echo "Job finished at: $(date)"
