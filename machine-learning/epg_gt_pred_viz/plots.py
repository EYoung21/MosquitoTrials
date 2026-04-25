"""Matplotlib helpers: ground truth vs predicted labels on EPG waveforms."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _label_sort_key(x: str) -> tuple[int, str]:
    s = str(x)
    if s.upper() == "NP":
        return (0, s)
    return (1, s)


def build_label_color_map(labels: Iterable[str]) -> dict[str, str]:
    uniq = sorted({str(x) for x in labels}, key=_label_sort_key)
    if not uniq:
        return {}
    try:
        import distinctipy

        colors = distinctipy.get_colors(len(uniq))
        hex_colors = [distinctipy.get_hex(c) for c in colors]
    except ImportError:
        cmap = plt.get_cmap("tab20")
        hex_colors = [mcolors.to_hex(cmap(i % 20) if len(uniq) > 1 else cmap(0)) for i in range(len(uniq))]
    return dict(zip(uniq, hex_colors))


def safe_label_filename(label: str) -> str:
    s = str(label).strip()
    if not s:
        return "EMPTY"
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    return s[:120] or "label"


def plot_epg_overview(
    time: np.ndarray,
    voltage: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_colors: dict[str, str],
) -> plt.Figure:
    """Three stacked panels: GT overlay, pred overlay, disagreement."""
    true_s = np.asarray(true_labels).astype(str)
    pred_s = np.asarray(pred_labels).astype(str)
    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    lo, hi = float(np.min(v)), float(np.max(v))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 9))

    axs[0].plot(t, v, color="black", linewidth=0.4)
    for lab, c in label_colors.items():
        axs[0].fill_between(t, lo, hi, where=(true_s == lab), color=c, alpha=0.45, label=str(lab))
    axs[0].set_title("EPG + ground truth (shaded by label)")
    axs[0].set_ylabel("Volts")
    axs[0].legend(loc="upper right", fontsize=7, ncol=4)

    axs[1].plot(t, v, color="black", linewidth=0.4)
    for lab, c in label_colors.items():
        axs[1].fill_between(t, lo, hi, where=(pred_s == lab), color=c, alpha=0.45, label=str(lab))
    axs[1].set_title("EPG + predicted (shaded by label)")
    axs[1].set_ylabel("Volts")
    axs[1].legend(loc="upper right", fontsize=7, ncol=4)

    axs[2].plot(t, v, color="black", linewidth=0.4)
    axs[2].fill_between(t, lo, hi, where=(pred_s != true_s), color="0.45", alpha=0.55)
    axs[2].set_title("EPG + disagreement (gray = pred ≠ true)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Volts")

    fig.tight_layout()
    return fig


def plot_epg_single_label(
    time: np.ndarray,
    voltage: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    label: str,
    label_color: str,
) -> plt.Figure | None:
    """
    For one semantic class: GT presence, predicted presence, and TP/FP/FN on the waveform.
    Returns None if this label never appears in true or pred for this segment.
    """
    true_s = np.asarray(true_labels).astype(str)
    pred_s = np.asarray(pred_labels).astype(str)
    lab = str(label)
    if not (np.any(true_s == lab) or np.any(pred_s == lab)):
        return None

    t = np.asarray(time, dtype=float)
    v = np.asarray(voltage, dtype=float)
    lo, hi = float(np.min(v)), float(np.max(v))

    tp = (true_s == lab) & (pred_s == lab)
    fn = (true_s == lab) & (pred_s != lab)
    fp = (true_s != lab) & (pred_s == lab)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 8))

    axs[0].plot(t, v, color="black", linewidth=0.4)
    axs[0].fill_between(t, lo, hi, where=(true_s == lab), color=label_color, alpha=0.5)
    axs[0].set_title(f'Label "{lab}" — ground truth regions')
    axs[0].set_ylabel("Volts")

    axs[1].plot(t, v, color="black", linewidth=0.4)
    axs[1].fill_between(t, lo, hi, where=(pred_s == lab), color=label_color, alpha=0.5)
    axs[1].set_title(f'Label "{lab}" — predicted regions')
    axs[1].set_ylabel("Volts")

    axs[2].plot(t, v, color="black", linewidth=0.4)
    axs[2].fill_between(t, lo, hi, where=tp, color="tab:green", alpha=0.55, label="TP")
    axs[2].fill_between(t, lo, hi, where=fn, color="tab:orange", alpha=0.55, label="FN")
    axs[2].fill_between(t, lo, hi, where=fp, color="tab:red", alpha=0.45, label="FP")
    axs[2].set_title(f'Label "{lab}" — TP / FN / FP on sequence')
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Volts")
    axs[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
