#!/bin/bash
#SBATCH --job-name=epg_boosting  # Name of the job
#SBATCH --output=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.out
#SBATCH --error=/home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/logs/%x_%j.err
#SBATCH --partition=gpu                # GPU partition
#SBATCH --gres=gpu:l40s:4              # Request 4 L40S GPUs (newer, well-supported)
#SBATCH --cpus-per-task=64             # Request 64 CPU cores (max for L40S nodes)
#SBATCH --mem=480G                     # Request 480GB memory (safe margin below 505GB max)
#SBATCH --time=7-00:00:00              # Max runtime: 7 days
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


# Run your Python script with Optuna hyperparameter optimization for Gradient Boosting
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/eyoung4-swat/hmc-epg-project/machine-learning/mosquito/eliIntroLabsResults/tarsalis_results_boosting --model_path boosting.py --model_name=boosting --optuna

echo "Job finished at: $(date)"

