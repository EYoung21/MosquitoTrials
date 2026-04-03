from omegaconf import DictConfig
from optuna.trial import Trial

def configure(cfg: DictConfig, trial: Trial) -> None:
    # already sampled in hydra.sweeper.params
    features = trial.params["model.model.features"]
    bottleneck_type = cfg.model.model.bottleneck_type
    dropout_is_zero = trial.params["model.model.dropout_is_zero"]
    weight_decay_is_zero = trial.params["train.weight_decay_is_zero"]

    # tie embed_dim to features
    trial.suggest_int("model.model.embed_dim", features, features)

    # conditional dropout
    if dropout_is_zero:
        trial.suggest_float("model.model.dropout_rate", 0.0, 0.0)
    else:
        trial.suggest_float("model.model.dropout_rate", 1e-2, 0.5, log=True)

    # conditional weight decay
    if weight_decay_is_zero:
        trial.suggest_float("train.weight_decay", 0.0, 0.0)
    else:
        trial.suggest_float("train.weight_decay", 1e-8, 1e-3, log=True)

    # only search transformer settings for attention bottlenecks
    if bottleneck_type in {"attention", "windowed_attention"}:
        trial.suggest_int("model.model.transformer_window_size", 100, 400, step=100)
        trial.suggest_int("model.model.transformer_layers", 1, 3, step=1)
        trial.suggest_categorical("model.model.transformer_nhead", [4, 8, 16])
    else:
        # pin them when bottleneck is not attention-based
        trial.suggest_int("model.model.transformer_window_size", -1, -1)
        trial.suggest_int("model.model.transformer_layers", 0, 0)
        trial.suggest_categorical("model.model.transformer_nhead", [1])