"""
GUI wrapper for the mosquito Random Forest model (samchanrf).
Adapts scido probe format (voltage, labels) to the RF model's expected format
(post_rect, resistance, voltage, current) and provides predict(probes, return_logits=True)
for the Labeler pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import torch

# Import the RF model from the machine-learning repo
_repo_root = Path(__file__).resolve().parents[4]
_ml_mosquito = _repo_root / "machine-learning" / "mosquito"
if _ml_mosquito.exists() and str(_ml_mosquito) not in sys.path:
    sys.path.insert(0, str(_ml_mosquito))
try:
    from samchanrf import Model as _RFModel
except ImportError:
    _RFModel = None


def _adapt_probes_for_rf(probes):
    """
    Convert GUI probes (DataFrames with time, voltage, labels) to the format
    expected by samchanrf: post_rect (waveform), resistance, voltage (scalar), current.
    """
    adapted = []
    for probe in probes:
        p = probe.copy()
        # RF uses "post_rect" as the waveform column; GUI has "voltage"
        if "post_rect" not in p.columns:
            p["post_rect"] = p["voltage"].values
        # Scalar metadata required by transform_data (used as constant per probe)
        if "resistance" not in p.columns:
            p["resistance"] = 1e9
        if "current" not in p.columns:
            p["current"] = "AC"
        # RF expects probe["voltage"].values[0] as a scalar (feature), not the time series
        if "voltage" in p.columns and (np.issubdtype(p["voltage"].dtype, np.number) and p["voltage"].size > 1):
            p["voltage"] = float(np.mean(p["voltage"].values))
        elif "voltage" not in p.columns or pd.isna(p["voltage"].iloc[0]):
            p["voltage"] = 0.0
        # Ensure labels exist for transform_data(training=False) path (not used but column may be checked)
        if "labels" not in p.columns:
            p["labels"] = "N"
        adapted.append(p)
    return adapted


class Model:
    """
    Scido-facing model that wraps the mosquito RF (samchanrf). Exposes
    load(path), predict(probes, return_logits=False), and inv_label_map
    so the Labeler can use it like the UNet models.
    """

    def __init__(self, save_path=None, trial=None):
        if _RFModel is None:
            raise RuntimeError(
                "Mosquito RF model could not be imported. "
                "Ensure machine-learning/mosquito/samchanrf.py is available."
            )
        self._rf = _RFModel(save_path=save_path, trial=trial)
        self.inv_label_map = None  # Set in load() from RF classes_

    def load(self, path=None):
        models_dir = Path(__file__).resolve().parent
        if path is None:
            path = models_dir / "rf_mosquito_pickle"
        else:
            path = Path(path)
            if not path.is_absolute():
                # Resolve relative to gui dir (e.g. "models/rf_mosquito_pickle" -> gui/models/rf_mosquito_pickle)
                gui_dir = models_dir.parent
                path = gui_dir / path
        if not path.exists():
            raise FileNotFoundError(
                f"RF model pickle not found: {path}. "
                "Train the model with machine-learning/mosquito/samchanrf.py and save the pickle, "
                "or place it at software/cs/gui/models/rf_mosquito_pickle"
            )
        with open(path, "rb") as f:
            self._rf.model = pickle.load(f)
        # Label map from trained classifier (index -> label string)
        self.inv_label_map = {i: c for i, c in enumerate(self._rf.model.classes_)}

    def predict(self, probes, return_logits=False):
        adapted = _adapt_probes_for_rf(probes)
        predictions = self._rf.predict(adapted)  # list of 1d arrays of label strings

        if not return_logits:
            return predictions

        # Build logits per probe: (1, num_classes, length) for postprocess_smooth
        num_classes = len(self._rf.model.classes_)
        label_to_idx = {c: i for i, c in enumerate(self._rf.model.classes_)}
        logits_list = []
        for pred_arr in predictions:
            n = len(pred_arr)
            # One-hot from predicted labels (indices)
            indices = np.array([label_to_idx.get(str(p), 0) for p in pred_arr], dtype=np.int64)
            logit = np.zeros((num_classes, n), dtype=np.float32)
            logit[indices, np.arange(n)] = 1.0
            logits_list.append(torch.tensor(logit).unsqueeze(0))  # (1, num_classes, n)
        return predictions, logits_list
