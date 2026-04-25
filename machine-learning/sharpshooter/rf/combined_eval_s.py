from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
)

# Ensure the sharpshooter directory (containing model_evaluation.py) is on sys.path
this_file = Path(__file__).resolve()
project_root = this_file.parents[1]  # .../sharpshooter
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also ensure the rf directory is importable as a plain module path
rf_dir = this_file.parent  # .../sharpshooter/rf
if str(rf_dir) not in sys.path:
    sys.path.insert(0, str(rf_dir))

from model_evaluation import DataImport, plot_labels
from rf_samchan_labels_combined import Model as CombinedModel


DATA_PATH = "/data/labs/hopelab/epg/epg_data/sharpshooter_parquet"

EXCLUDE = {
    "a01",
    "a02",
    "a03",
    "a10",
    "a15",
    "b01",
    "b02",
    "b04",
    "b07",
    "b12",
    "b188",
    "b202",
    "b206",
    "b208",
    "c046",
    "c07",
    "c09",
    "c10",
    "d01",
    "d03",
    "d056",
    "d058",
    "d12",
}

SHARPSHOOTER_DIR = Path("/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter")
SAVE_PATH = SHARPSHOOTER_DIR / "rf" / "combined_eval_s"
MODEL_NAME = "rf"


def generate_report_mosquito_style(
    test_data,
    predicted_labels,
    test_names,
    save_path: Path,
    model_name: str,
    fold: int,
):
    labels_true = []
    labels_pred = []
    for df, preds in zip(test_data, predicted_labels):
        labels_true.extend(df["labels"].values)
        labels_pred.extend(preds)

    labels_true = np.asarray(labels_true).astype(str)
    labels_pred = np.asarray(labels_pred).astype(str)

    labels = sorted(np.unique(labels_true))
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels_true,
        labels_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    metrics = {}
    for label, p, r, f in zip(labels, precision, recall, fscore):
        metrics[f"precision_{label}"] = p
        metrics[f"recall_{label}"] = r
        metrics[f"fscore_{label}"] = f

    out_dataframe = pd.DataFrame([metrics])
    out_dataframe["accuracy"] = accuracy_score(labels_true, labels_pred)

    ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, normalize="true")
    plt.savefig(save_path / f"{model_name}_ConfusionMatrix_Fold{fold}.png")
    plt.close()

    for df, preds, name in zip(test_data, predicted_labels, test_names):
        fig = plot_labels(
            df["time"],
            df["voltage"],
            df["labels"].values,
            np.asarray(preds),
        )
        stem = Path(str(name)).name
        fig.savefig(save_path / f"{model_name}_{stem}_Fold{fold}.png")
        plt.close(fig)

    print(f"Fold {fold} Overall Accuracy: {out_dataframe['accuracy'].iloc[0]}")
    return labels_true.tolist(), labels_pred.tolist(), out_dataframe


def build_model(save_path: str | None = None) -> CombinedModel:
    model = CombinedModel(save_path=save_path)

    # Best params from labels-combined trial 98
    model.chunk_seconds = 25
    model.num_freqs = 11
    model.num_estimators = 1024
    model.max_depth = 128
    model.max_features = 0.2006656546810392
    model.overlap = 0.5474607646696771
    model.max_lag = 10

    model.chunk_size = model.chunk_seconds * model.sample_rate
    sigma = model.chunk_size / 6
    center = (model.chunk_size - 1) / 2
    window_weight = np.exp(-0.5 * ((np.arange(model.chunk_size) - center) / sigma) ** 2)
    model.window_weight = window_weight / np.sum(window_weight)

    return model


def main() -> None:
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    print("Loading sharpshooter data for combined-label RF evaluation (combined_eval_s)...")
    data = DataImport(
        DATA_PATH,
        filetype=".parquet",
        exclude=EXCLUDE,
        folds=5,
        binary=False,
        coarse_first_letter_labels=True,
    )

    summary_frames: list[pd.DataFrame] = []
    all_true: list[str] = []
    all_pred: list[str] = []

    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        print(f"=== Evaluating combined-label RF (trial 98 params), fold {fold} ===")

        train_dfs = [data.df_list[i] for i in train_index]
        test_dfs = [data.df_list[i] for i in test_index]

        train_probes, _ = data.get_probes(train_dfs)
        test_probes, test_names = data.get_probes(test_dfs)

        model = build_model(save_path=str(SAVE_PATH))
        model.train(train_probes, test_probes, fold)
        preds = model.predict(test_probes)

        true, pred, stats = generate_report_mosquito_style(
            test_probes,
            preds,
            test_names,
            SAVE_PATH,
            MODEL_NAME,
            fold,
        )

        summary_frames.append(stats)
        all_true.extend(true)
        all_pred.extend(pred)

    if summary_frames:
        summary_df = pd.concat(summary_frames)
        summary_df.to_csv(SAVE_PATH / f"{MODEL_NAME}_SummaryStats.csv")
        print(f"Saved per-fold summary statistics to {SAVE_PATH / f'{MODEL_NAME}_SummaryStats.csv'}")

    pd.DataFrame({"labels_true": all_true, "labels_pred": all_pred}).to_csv(
        SAVE_PATH / f"{MODEL_NAME}_allpredictions.csv"
    )

    if all_true and all_pred:
        ConfusionMatrixDisplay.from_predictions(all_true, all_pred, normalize="true")
        plt.savefig(SAVE_PATH / f"{MODEL_NAME}_OverallConfusionMatrix.png")
        plt.close()

    print("Combined-label RF evaluation complete (combined_eval_s).")


if __name__ == "__main__":
    main()

