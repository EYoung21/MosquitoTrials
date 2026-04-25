import argparse
import numpy as np
import optuna

from model_eval import DataImport, optuna_objective


def main():
    parser = argparse.ArgumentParser(
        prog="Mosquito Binary NP/P Optuna",
        description="Run Optuna for mosquito RF with labels collapsed to NP vs P.",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--optuna", action="store_true")
    args = parser.parse_args()

    if not args.optuna:
        raise ValueError("This entrypoint is intended for Optuna runs only. Pass --optuna.")

    print("Loading Data...")
    data = DataImport(args.data_path, 5)

    # Collapse all probing labels to P and keep NP as NP.
    for df in data.raw_dfs:
        labels = df["labels"].astype(str).str.upper()
        df["labels"] = np.where(labels == "NP", "NP", "P")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda x: optuna_objective(data, args, x), n_trials=100, show_progress_bar=True)

    print("Best trial:")
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
