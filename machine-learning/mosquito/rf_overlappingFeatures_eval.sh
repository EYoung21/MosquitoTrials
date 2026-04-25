#!/bin/bash
#SBATCH --job-name=epg_rf_overlapping_eval  # Name of the job
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.err
#SBATCH --partition=gpu                # GPU partition
#SBATCH --gres=gpu:l40s:1              # Request 1 L40S GPU (evaluation doesn't need 4)
#SBATCH --cpus-per-task=16             # Request 16 CPU cores
#SBATCH --mem=120G                     # Request 120GB memory
#SBATCH --time=1-00:00:00              # Max runtime: 1 day
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


# Run your Python script for model evaluation (WITHOUT --optuna flag)
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/tarsalis_results_overlapping_eval --model_path rf_overlappingFeatures.py --model_name=rf_overlappingFeatures

echo "Job finished at: $(date)"
