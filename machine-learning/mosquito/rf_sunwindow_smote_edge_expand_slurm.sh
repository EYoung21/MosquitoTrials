#!/bin/bash
#SBATCH --job-name=epg_rf_smote_edge_expand
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

# Optuna rerun with expanded search at former boundaries:
#   augment_factor, chunk_seconds (now 0.5s steps up to window), smote_k_neighbors (1–20).
# Logs: rf_sunwindow_smote_edge_expand_optuna.txt (model_name below).
# --augment so augment_factor affects training (was omitted in the original smote slurm script).

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
    --save_path /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/tarsalis_results_smote_edge_expand \
    --model_path rf_sunwindow_smote.py --model_name=rf_sunwindow_smote_edge_expand --optuna --augment

echo "Job finished at: $(date)"
