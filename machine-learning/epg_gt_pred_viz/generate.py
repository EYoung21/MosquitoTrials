"""
Train one CV fold of the RF pipeline and save EPG sequence plots:
overview (GT / pred / error) plus one figure per label (TP/FP/FN).

Run from anywhere, e.g.:
  cd /home/eyoung4-swat/hmc-epg-project/machine-learning/epg_gt_pred_viz
  uv run --project ../mosquito python generate.py --dataset mosquito --max-probes 8
  uv run --project ../sharpshooter python generate.py --dataset sharpshooter --max-probes 8
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
MOSQUITO_DIR = ML_ROOT / "mosquito"
SHARPSHOOTER_DIR = ML_ROOT / "sharpshooter"
DEFAULT_OUT = ML_ROOT / "epg_gt_pred_viz" / "output"

SH_EXCLUDE = {
    "a01", "a02", "a03", "a10", "a15",
    "b01", "b02", "b04", "b07", "b12", "b188", "b202", "b206", "b208",
    "c046", "c07", "c09", "c10",
    "d01", "d03", "d056", "d058", "d12",
}


def _dynamic_import(module_path: Path):
    spec = importlib.util.spec_from_file_location("user_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_mosquito(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(MOSQUITO_DIR))
    from model_eval import DataImport  # noqa: E402

    data = DataImport(args.data_path, args.folds)
    if args.binary_np_p:
        for df in data.raw_dfs:
            labels = df["labels"].astype(str).str.upper()
            df["labels"] = np.where(labels == "NP", "NP", "P")

    fold = args.fold
    train_index, test_index = data.cross_val_iter[fold]
    train_dfs = [data.raw_dfs[i] for i in train_index]
    test_dfs = [data.raw_dfs[i] for i in test_index]
    train_probes, _ = data.get_probes(train_dfs)
    test_probes, test_names = data.get_probes(test_dfs)

    if args.max_probes is not None:
        test_probes = test_probes[: args.max_probes]
        test_names = test_names[: args.max_probes]

    model_path = Path(args.model_path)
    if not model_path.is_file():
        model_path = MOSQUITO_DIR / args.model_path
    model_mod = _dynamic_import(model_path)
    model = model_mod.Model(save_path=str(args.save_path), trial=None)
    model.train(train_probes, test_probes, fold)
    predicted_labels = model.predict(test_probes)

    out_root = Path(args.save_path) / args.run_name
    overview_dir = out_root / "overview"
    label_dir = out_root / "by_label"

    from plots import build_label_color_map, plot_epg_overview, plot_epg_single_label, safe_label_filename, save_figure

    all_labels: set[str] = set()
    for df, preds in zip(test_probes, predicted_labels):
        all_labels.update(np.asarray(df["labels"].values).astype(str))
        all_labels.update(np.asarray(preds).astype(str))
    label_colors = build_label_color_map(all_labels)

    for df, preds, name in zip(test_probes, predicted_labels, test_names):
        stem = Path(str(name)).stem
        t = df["time"].values
        v = df[args.voltage_column].values
        y = df["labels"].values
        p = np.asarray(preds).astype(str)

        fig_o = plot_epg_overview(t, v, y, p, label_colors)
        save_figure(fig_o, overview_dir / f"{stem}_fold{fold}.png")

        probe_labels = sorted(set(np.unique(y)).union(set(np.unique(p))), key=str)
        for lab in probe_labels:
            fig_l = plot_epg_single_label(t, v, y, p, str(lab), label_colors.get(str(lab), "#888888"))
            if fig_l is None:
                continue
            ld = label_dir / safe_label_filename(str(lab))
            save_figure(fig_l, ld / f"{stem}_fold{fold}.png")


def run_sharpshooter(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(SHARPSHOOTER_DIR))
    from model_evaluation import DataImport  # noqa: E402

    data = DataImport(
        args.data_path,
        filetype=args.filetype,
        exclude=SH_EXCLUDE,
        folds=args.folds,
        binary=args.binary,
        coarse_first_letter_labels=args.coarse_first_letter_labels,
    )

    fold = args.fold
    train_index, test_index = data.cross_val_iter[fold]
    train_data = [data.df_list[i] for i in train_index]
    test_data = [data.df_list[i] for i in test_index]

    if not getattr(data, "binary", False):
        train_data, _ = data.get_probes(train_data)
        test_data, test_names = data.get_probes(test_data)
    else:
        test_names = [Path(df.attrs["file"]).stem for df in test_data]

    if args.max_probes is not None:
        test_data = test_data[: args.max_probes]
        test_names = test_names[: args.max_probes]

    model_path = Path(args.model_path)
    if not model_path.is_file():
        model_path = SHARPSHOOTER_DIR / args.model_path
    model_mod = _dynamic_import(model_path)
    model = model_mod.Model(save_path=str(args.save_path), trial=None)
    model.train(train_data)
    predicted_labels = model.predict(test_data)

    out_root = Path(args.save_path) / args.run_name
    overview_dir = out_root / "overview"
    label_dir = out_root / "by_label"

    from plots import build_label_color_map, plot_epg_overview, plot_epg_single_label, safe_label_filename, save_figure

    all_labels: set[str] = set()
    for df, preds in zip(test_data, predicted_labels):
        all_labels.update(np.asarray(df["labels"].values).astype(str))
        all_labels.update(np.asarray(preds).astype(str))
    label_colors = build_label_color_map(all_labels)

    for df, preds, name in zip(test_data, predicted_labels, test_names):
        stem = Path(str(name)).stem
        t = df["time"].values
        v = df["voltage"].values
        y = df["labels"].values
        p = np.asarray(preds).astype(str)

        fig_o = plot_epg_overview(t, v, y, p, label_colors)
        save_figure(fig_o, overview_dir / f"{stem}_fold{fold}.png")

        probe_labels = sorted(set(np.unique(y)).union(set(np.unique(p))), key=str)
        for lab in probe_labels:
            fig_l = plot_epg_single_label(t, v, y, p, str(lab), label_colors.get(str(lab), "#888888"))
            if fig_l is None:
                continue
            ld = label_dir / safe_label_filename(str(lab))
            save_figure(fig_l, ld / f"{stem}_fold{fold}.png")


def main() -> None:
    # Local imports (`plots`) resolve when this script's directory is on sys.path
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    p = argparse.ArgumentParser(description="EPG ground truth vs predicted sequence visualizations")
    p.add_argument("--dataset", choices=("mosquito", "sharpshooter"), required=True)
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Override data directory (defaults per dataset).",
    )
    p.add_argument("--model_path", type=str, default=None, help="Path to model .py (default per dataset).")
    p.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_OUT),
        help=f"Root output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Subfolder under save_path (default: mosquito_viz or sharpshooter_viz).",
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-probes", type=int, default=12, dest="max_probes")
    p.add_argument(
        "--voltage-column",
        type=str,
        default="pre_rect",
        dest="voltage_column",
        help="Mosquito only: waveform column (default pre_rect, matches model_eval plots).",
    )
    p.add_argument(
        "--binary-np-p",
        action="store_true",
        dest="binary_np_p",
        help="Mosquito only: collapse labels to NP vs P before probe extraction.",
    )
    p.add_argument("--binary", action="store_true", help="Sharpshooter: NP vs P mode.")
    p.add_argument(
        "--coarse-first-letter-labels",
        action="store_true",
        dest="coarse_first_letter_labels",
        help="Sharpshooter: collapse labels to first letter (match combined-label RF).",
    )
    p.add_argument("--filetype", type=str, default=".parquet", help="Sharpshooter file filter.")

    args = p.parse_args()

    if args.data_path is None:
        if args.dataset == "mosquito":
            args.data_path = "/data/labs/hopelab/epg/tarsalis_data_clean"
        else:
            args.data_path = "/data/labs/hopelab/epg/epg_data/sharpshooter_parquet"

    if args.model_path is None:
        if args.dataset == "mosquito":
            args.model_path = str(MOSQUITO_DIR / "rf_sunwindow_smote.py")
        else:
            args.model_path = str(SHARPSHOOTER_DIR / "rf" / "rf_samchan_shortfix.py")

    if args.run_name is None:
        args.run_name = f"{args.dataset}_viz_fold{args.fold}"

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    if args.dataset == "mosquito":
        run_mosquito(args)
    else:
        run_sharpshooter(args)

    print(f"Wrote figures under {Path(args.save_path) / args.run_name}")


if __name__ == "__main__":
    main()
