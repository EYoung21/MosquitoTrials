#!/bin/bash
#SBATCH --job-name=epg_rf_smote_eval  # Name of the job - evaluation with optimized hyperparameters
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.err
#SBATCH --partition=gpu                # GPU partition
#SBATCH --gres=gpu:l40s:2              # Request 2 L40S GPUs (less than optuna run)
#SBATCH --cpus-per-task=32             # Request 32 CPU cores
#SBATCH --mem=240G                     # Request 240GB memory
#SBATCH --time=2-00:00:00              # Max runtime: 2 days
#SBATCH --nodes=1                      # Single node

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  # NOTE: Requires outbound internet; if your cluster blocks it,
  # ask your admin to provide a uv module or preinstall it in your image.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# Create log directory if it doesn't exist
mkdir -p /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs

# Print info for debugging
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"


# Run full evaluation with optimized hyperparameters (NO --optuna flag)
# Using optimized hyperparameters from Trial 95 (F1=0.5444)
# augment_factor=1 means no augmentation needed
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/tarsalis_results_smote_optimized --model_path rf_sunwindow_smote.py --model_name=rf_sunwindow_smote_optimized

echo "Job finished at: $(date)"
