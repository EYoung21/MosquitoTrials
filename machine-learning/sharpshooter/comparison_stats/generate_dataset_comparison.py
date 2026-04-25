from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MOSQUITO_DIR = Path("/data/labs/hopelab/epg/tarsalis_data_clean")
SHARPSHOOTER_DIR = Path("/data/labs/hopelab/epg/epg_data/sharpshooter_parquet")
OUT_DIR = Path("/home/eyoung4-swat/hmc-epg-project/machine-learning/sharpshooter/comparison_stats")


@dataclass
class DatasetConfig:
    name: str
    file_pattern: str
    non_probing_labels: set[str]


def _read_labeled_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if "labels" not in df.columns:
        raise ValueError(f"Missing labels column in {path}")
    out = df.copy()
    out["labels"] = out["labels"].astype(str).str.strip()
    return out


def _probe_lengths(labels: np.ndarray, non_probing_labels: set[str]) -> list[int]:
    mask = ~np.isin(np.char.upper(labels.astype(str)), list(non_probing_labels))
    lengths: list[int] = []
    run = 0
    for val in mask:
        if val:
            run += 1
        elif run > 0:
            lengths.append(run)
            run = 0
    if run > 0:
        lengths.append(run)
    return lengths


def _safe_quantiles(values: np.ndarray, qs: Iterable[float]) -> list[float]:
    if values.size == 0:
        return [float("nan") for _ in qs]
    return [float(np.quantile(values, q)) for q in qs]


def build_stats(base_dir: Path, cfg: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(base_dir.glob(cfg.file_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for {cfg.name} in {base_dir} with {cfg.file_pattern}")

    label_counter: dict[str, int] = {}
    per_file_rows: list[dict[str, float | str | int]] = []
    probe_lengths_all: list[int] = []
    total_samples = 0

    for fpath in files:
        df = _read_labeled_df(fpath)
        labels = df["labels"].to_numpy(dtype=str)
        total_samples += len(labels)
        uniq, counts = np.unique(labels, return_counts=True)
        for lab, count in zip(uniq, counts):
            label_counter[lab] = label_counter.get(lab, 0) + int(count)

        probe_lengths = _probe_lengths(labels, cfg.non_probing_labels)
        probe_lengths_all.extend(probe_lengths)
        q25, q50, q75, q95 = _safe_quantiles(np.asarray(probe_lengths, dtype=float), [0.25, 0.50, 0.75, 0.95])
        per_file_rows.append(
            {
                "dataset": cfg.name,
                "file": fpath.name,
                "samples": int(len(labels)),
                "unique_labels": int(len(np.unique(labels))),
                "num_probes": int(len(probe_lengths)),
                "probe_len_mean": float(np.mean(probe_lengths)) if probe_lengths else float("nan"),
                "probe_len_p50": q50,
                "probe_len_p95": q95,
            }
        )

    label_df = (
        pd.DataFrame(
            {
                "dataset": cfg.name,
                "label": list(label_counter.keys()),
                "count": list(label_counter.values()),
            }
        )
        .sort_values(["count", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    label_df["proportion"] = label_df["count"] / max(1, label_df["count"].sum())

    probe_np = np.asarray(probe_lengths_all, dtype=float)
    q25, q50, q75, q95 = _safe_quantiles(probe_np, [0.25, 0.50, 0.75, 0.95])
    summary_df = pd.DataFrame(
        [
            {
                "dataset": cfg.name,
                "num_files": len(files),
                "total_samples": int(total_samples),
                "num_unique_labels": int(label_df["label"].nunique()),
                "num_total_probes": int(len(probe_lengths_all)),
                "probe_len_mean": float(np.mean(probe_np)) if probe_np.size else float("nan"),
                "probe_len_std": float(np.std(probe_np)) if probe_np.size else float("nan"),
                "probe_len_p25": q25,
                "probe_len_p50": q50,
                "probe_len_p75": q75,
                "probe_len_p95": q95,
            }
        ]
    )

    per_file_df = pd.DataFrame(per_file_rows).sort_values("file").reset_index(drop=True)
    return summary_df, label_df, per_file_df


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([header, sep] + body)


def _to_markdown_report(summary: pd.DataFrame, labels: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Sharpshooter vs Mosquito Dataset Comparison")
    lines.append("")
    lines.append("## High-level Summary")
    lines.append(_markdown_table(summary))
    lines.append("")

    for dataset in summary["dataset"].tolist():
        lines.append(f"## Label Distribution: {dataset}")
        sub = labels[labels["dataset"] == dataset][["label", "count", "proportion"]].copy()
        sub["proportion"] = sub["proportion"].map(lambda x: f"{x:.4f}")
        lines.append(_markdown_table(sub))
        lines.append("")

    return "\n".join(lines)


def _save_visualizations(summary: pd.DataFrame, labels: pd.DataFrame, out_dir: Path) -> None:
    # 1) Dataset size comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["dataset"], summary["total_samples"], color=["#4e79a7", "#f28e2b"])
    ax.set_title("Total Samples by Dataset")
    ax.set_ylabel("Samples")
    ax.ticklabel_format(axis="y", style="plain")
    fig.tight_layout()
    fig.savefig(out_dir / "viz_total_samples_comparison.png", dpi=180)
    plt.close(fig)

    # 2) Probe length distribution comparison via boxplot
    probe_df = pd.read_csv(out_dir / "dataset_per_file_stats.csv")
    plot_probe = probe_df.dropna(subset=["probe_len_p50"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [plot_probe.loc[plot_probe["dataset"] == d, "probe_len_p50"].values for d in summary["dataset"]]
    ax.boxplot(data, tick_labels=summary["dataset"].tolist(), showfliers=False)
    ax.set_title("Per-file Median Probe Length Comparison")
    ax.set_ylabel("Probe length (samples)")
    fig.tight_layout()
    fig.savefig(out_dir / "viz_probe_length_boxplot.png", dpi=180)
    plt.close(fig)

    # 3) High-level probe length stats (mean/p50/p95)
    long_rows = []
    for _, row in summary.iterrows():
        long_rows.extend(
            [
                {"dataset": row["dataset"], "metric": "mean", "value": row["probe_len_mean"]},
                {"dataset": row["dataset"], "metric": "p50", "value": row["probe_len_p50"]},
                {"dataset": row["dataset"], "metric": "p95", "value": row["probe_len_p95"]},
            ]
        )
    long_df = pd.DataFrame(long_rows)
    metrics = ["mean", "p50", "p95"]
    datasets = summary["dataset"].tolist()
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ds in enumerate(datasets):
        vals = [float(long_df[(long_df["dataset"] == ds) & (long_df["metric"] == m)]["value"].iloc[0]) for m in metrics]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=ds)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Probe Length Stats Comparison")
    ax.set_ylabel("Samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "viz_probe_length_stats_comparison.png", dpi=180)
    plt.close(fig)

    # 4) Label distribution percentages side-by-side (top labels per dataset)
    label_plot = labels.copy()
    label_plot["pct"] = label_plot["proportion"] * 100.0
    top_m = label_plot[label_plot["dataset"] == "mosquito"].sort_values("pct", ascending=False).head(12)
    top_s = label_plot[label_plot["dataset"] == "sharpshooter"].sort_values("pct", ascending=False).head(12)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
    axes[0].barh(top_m["label"][::-1], top_m["pct"][::-1], color="#4e79a7")
    axes[0].set_title("Mosquito Label Distribution (%)")
    axes[0].set_xlabel("Percent of samples")
    axes[1].barh(top_s["label"][::-1], top_s["pct"][::-1], color="#f28e2b")
    axes[1].set_title("Sharpshooter Label Distribution (%)")
    axes[1].set_xlabel("Percent of samples")
    fig.tight_layout()
    fig.savefig(out_dir / "viz_label_distribution_percentages.png", dpi=180)
    plt.close(fig)

    # 5) Binary NP/P-style comparison percentages
    def _binary_pct(df: pd.DataFrame, dataset_name: str) -> tuple[float, float]:
        sub = df[df["dataset"] == dataset_name].copy()
        sub["label_u"] = sub["label"].astype(str).str.upper()
        if dataset_name == "mosquito":
            np_mask = sub["label_u"].eq("NP")
        else:
            np_mask = sub["label_u"].isin(["N", "Z", "NP"])
        np_count = sub.loc[np_mask, "count"].sum()
        p_count = sub.loc[~np_mask, "count"].sum()
        total = max(1, np_count + p_count)
        return 100.0 * np_count / total, 100.0 * p_count / total

    m_np, m_p = _binary_pct(labels, "mosquito")
    s_np, s_p = _binary_pct(labels, "sharpshooter")
    comp = pd.DataFrame(
        [
            {"dataset": "mosquito", "NP_pct": m_np, "P_pct": m_p},
            {"dataset": "sharpshooter", "NP_pct": s_np, "P_pct": s_p},
        ]
    )
    comp.to_csv(out_dir / "dataset_binary_np_p_percentages.csv", index=False)

    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, comp["NP_pct"], label="NP-like %")
    ax.bar(x, comp["P_pct"], bottom=comp["NP_pct"], label="P-like %")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["dataset"])
    ax.set_ylabel("Percent of samples")
    ax.set_title("Binary NP/P-style Label Composition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "viz_binary_np_p_percentages_stacked.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mosquito_cfg = DatasetConfig(
        name="mosquito",
        file_pattern="*.csv",
        non_probing_labels={"NP"},
    )
    sharpshooter_cfg = DatasetConfig(
        name="sharpshooter",
        file_pattern="*.parquet",
        non_probing_labels={"N", "Z", "NP"},
    )

    m_summary, m_labels, m_files = build_stats(MOSQUITO_DIR, mosquito_cfg)
    s_summary, s_labels, s_files = build_stats(SHARPSHOOTER_DIR, sharpshooter_cfg)

    summary = pd.concat([m_summary, s_summary], ignore_index=True)
    labels = pd.concat([m_labels, s_labels], ignore_index=True)
    per_file = pd.concat([m_files, s_files], ignore_index=True)

    summary.to_csv(OUT_DIR / "dataset_summary_comparison.csv", index=False)
    labels.to_csv(OUT_DIR / "dataset_label_distribution_comparison.csv", index=False)
    per_file.to_csv(OUT_DIR / "dataset_per_file_stats.csv", index=False)

    _save_visualizations(summary, labels, OUT_DIR)

    report = _to_markdown_report(summary, labels)
    (OUT_DIR / "dataset_comparison_report.md").write_text(report)

    print("Wrote:")
    print(OUT_DIR / "dataset_summary_comparison.csv")
    print(OUT_DIR / "dataset_label_distribution_comparison.csv")
    print(OUT_DIR / "dataset_per_file_stats.csv")
    print(OUT_DIR / "dataset_binary_np_p_percentages.csv")
    print(OUT_DIR / "viz_total_samples_comparison.png")
    print(OUT_DIR / "viz_probe_length_boxplot.png")
    print(OUT_DIR / "viz_probe_length_stats_comparison.png")
    print(OUT_DIR / "viz_label_distribution_percentages.png")
    print(OUT_DIR / "viz_binary_np_p_percentages_stacked.png")
    print(OUT_DIR / "dataset_comparison_report.md")


if __name__ == "__main__":
    main()
