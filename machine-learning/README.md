To ML team: update this as you see fit!

This subrepository contains the ML development code for the waveform classifier included in [SCIDO](https://www.github.com/jomotham/tree/main/software). The most recent/best version of these models are packaged with SCIDO.

[Link to project Drive](https://drive.google.com/drive/folders/1IeiOQtImzPjvFvvDDb7daoPg8Atfd-Y5?usp=sharing)


# Installation
We recommend using [**uv**](https://docs.astral.sh/uv/) to manage this folder's environment and dependencies.  
If you prefer, you can use `pip` with a `venv`, but uv is simpler and fully reproducible.

### Setup with uv
From the `machine-learning` folder of the git repository (open this as your workspace in your editor):

```bash
uv sync --extra <compute-platform>
```
#### Valid compute platform options
- `cpu` – CPU-only (default for macOS)  
- `cu129` – CUDA 12.9 (Windows/Linux with compatible NVIDIA drivers)  
- `cu128` – CUDA 12.8  
- `cu126` – CUDA 12.6  

To check which CUDA version your system supports (Windows/Linux only; **macOS does not support CUDA**):

```bash
nvcc --version
```


### Setup with pip (alternative)
If you’re not using uv, you can install the base dependencies directly:

```bash
pip install .
```
 
**Important:** pip will **not** automatically install PyTorch.  
You must install the correct wheel for your system from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).  


# Weights & Biases + Weave

Both mosquito and sharpshooter log to W&B for experiment tracking and Weave for Optuna trace inspection.

**Setup:** create a `.env` file in `machine-learning/` with your [W&B API key](https://wandb.ai/settings#apikeys):
```bash
WANDB_API_KEY=<your key>
```
The SLURM scripts load this automatically.

**Projects:** `hmc-epg-mosquito` and `hmc-epg-sharpshooter` under `hopelab-swarthmore`.

**What's logged per fold:** train/val loss curves (epoch on x-axis), per-class precision/recall/F1 as a filterable table, confusion matrix image, and scalar macro-F1/accuracy. The `overall` summary run aggregates across all folds.

**Optuna:** mosquito uses nested CV (one study per outer fold); sharpshooter runs one global study across all folds and exits before the eval. After each trial, `log_optuna_trial_snapshot` logs each hyperparameter as its own column in Weave Traces