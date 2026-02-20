#!/bin/bash
#SBATCH --job-name=epg_job          # Name of the job
#SBATCH --output=/data/labs/hopelab/epg/logs/%x_%j.out
#SBATCH --error=/data/labs/hopelab/epg/logs/%x_%j\.err
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=8           # Request 8 CPU cores
#SBATCH --mem=16G                   # Request 16 GB memory
#SBATCH --time=24:00:00             # Max runtime (adjust as needed)
#SBATCH --partition=gpu             # Use a GPU partition if applicable

if ! command -v uv >/dev/null 2>&1; then
  echo "[info] uv not found in PATH; attempting local install to ~/.local/bin"
  # NOTE: Requires outbound internet; if your cluster blocks it,
  # ask your admin to provide a uv module or preinstall it in your image.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# Create log directory if it doesn’t exist
mkdir -p /data/labs/hopelab/epg/logs

# Print info for debugging
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "Running job ID: $SLURM_JOB_ID"

# ==== WANDB CONFIGURATION ====

# To store wandb logs in the run directory instead of the project root:
export WANDB_DIR="/data/labs/hopelab/epg/logs"

#echo "Random forest evaluation"
#uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/nested_model_evaluation_forest --model_path rf.py --model_name=rf --optuna # --post_process v
# # --optuna

# Run your Python script
echo "UNET attention evaluation"
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet_attention --model_path unet.py --model_name=unet --attention  --optuna # --post_process v
# --optuna

echo "UNET evaluation"
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_unet --model_path unet.py --model_name=unet  --optuna 

echo "TCN evaluation"
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_tcn --model_path tcn.py --model_name=tcn  --optuna # --post_process v
echo "Transformer evaluation"
uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
    --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/vnested_model_evaluation_transformer --model_path transformer.py --model_name=transformer  --optuna # --post_process v



# uv run --extra cu129 model_eval.py --data_path /data/labs/hopelab/epg/tarsalis_data_clean \
#     --save_path /home/ghope1-swat/EPG-Project/hmc-epg-project/machine-learning/mosquito/model_evaluation_crf --model_path unet_crf.py --model_name=unet --attention


echo "Job finished at: $(date)"
