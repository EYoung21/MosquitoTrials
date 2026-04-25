import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
from scipy.signal import hilbert

warnings.simplefilter(action="ignore", category=FutureWarning)


class Model:
    def __init__(self, save_path=None, trial=None):
        # Seed defaults from prior best rf_samchan trial (48)
        self.chunk_seconds = 9
        self.num_estimators = 128
        self.max_depth = 32
        self.max_features = 0.5546751344739631
        self.num_freqs = 10
        self.sample_rate = 100
        self.chunk_size = self.chunk_seconds * self.sample_rate
        self.waveform_type = "voltage"
        self.random_state = 42
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"
        self.overlap = 0.47787211126294354
        self.max_lag = 8

        if trial:
            self.chunk_seconds = trial.suggest_int("chunk_seconds", 1, 25)
            self.num_freqs = trial.suggest_int("num_freqs", 1, 25)
            self.num_estimators = trial.suggest_categorical("num_estimators", [8, 16, 32, 64, 128, 256, 512, 1024])
            self.max_depth = trial.suggest_categorical("max_depth", [4, 8, 16, 32, 64, 128, 256])
            self.max_features = trial.suggest_float("max_features", 0.2, 1.0)
            self.chunk_size = self.chunk_seconds * self.sample_rate
            self.overlap = trial.suggest_float("overlap", 0.25, 0.85)
            self.max_lag = trial.suggest_int("max_lag", 3, 30)

        self.model = None

        sigma = self.chunk_size / 6
        center = (self.chunk_size - 1) / 2
        self.window_weight = np.exp(-0.5 * ((np.arange(self.chunk_size) - center) / sigma) ** 2)
        self.window_weight /= np.sum(self.window_weight)

    def transform_data(self, probes, training=True):
        transformed_probes = []
        for probe in probes:
            n_samples = len(probe)
            if n_samples == 0:
                if not training:
                    transformed_probes.append(pd.DataFrame())
                continue

            data_array = probe[self.waveform_type].values
            labels_array = probe["labels"].values if training else None
            increment = max(1, int(self.chunk_size * (1 - self.overlap)))
            start_indices = np.arange(0, max(n_samples - self.chunk_size + 1, 1), increment)
            n_windows = len(start_indices)

            freqs = np.fft.fftfreq(self.chunk_size, 1 / self.sample_rate)[1 : self.chunk_size // 2]
            columns = defaultdict(list)

            for start in start_indices:
                end = start + self.chunk_size
                if end <= n_samples:
                    window = data_array[start:end]
                    label_window = labels_array[start:end] if training else None
                else:
                    pad_size = end - n_samples
                    window = np.concatenate([data_array[start:], np.full(pad_size, data_array[-1])])
                    if training:
                        label_window = np.concatenate([labels_array[start:], np.full(pad_size, labels_array[-1])])

                fft_vals = np.abs(np.fft.fft(window))[1 : self.chunk_size // 2]
                fft_vals /= np.linalg.norm(fft_vals) + 1e-12
                peak_count = min(self.num_freqs, len(fft_vals))
                peak_indices = np.argsort(-fft_vals)[:peak_count]
                peak_freqs = freqs[peak_indices]
                if peak_count < self.num_freqs:
                    peak_freqs = np.pad(peak_freqs, (0, self.num_freqs - peak_count))
                for j in range(self.num_freqs):
                    columns[f"F{j}"].append(peak_freqs[j])

                envelope = np.abs(hilbert(window))
                columns["env_mean"].append(np.mean(envelope))
                columns["env_std"].append(np.std(envelope))
                columns["mean"].append(np.mean(window))
                columns["std"].append(np.std(window))
                columns["rms"].append(np.sqrt(np.mean(window**2)))

                for lag in range(1, self.max_lag + 1):
                    x = window[:-lag]
                    y = window[lag:]
                    cov = np.mean((x - np.mean(x)) * (y - np.mean(y)))
                    columns[f"autocorr_lag{lag}"].append(cov / (np.std(x) * np.std(y) + 1e-12))

                if training:
                    vals, counts = np.unique(label_window, return_counts=True)
                    columns["label"].append(vals[np.argmax(counts)])

            columns["resistance"] = [probe["resistance"].values[0]] * n_windows if "resistance" in probe.columns else [0.0] * n_windows
            columns["volts"] = [probe["voltage"].values[0]] * n_windows
            columns["current"] = [0 if ("current" in probe.columns and probe["current"].values[0] == "AC") else 1] * n_windows
            columns["position"] = np.arange(n_windows) / max(1, n_windows)
            transformed_probes.append(pd.DataFrame(columns).fillna(0))

        return transformed_probes

    def train(self, probes, test_data=None, fold=None):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        x_train = train.drop(["label"], axis=1)
        y_train = train["label"]
        self.model = RandomForestClassifier(
            n_estimators=self.num_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(x_train, y_train)

    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training=False)
        predictions = []
        for df, raw_probe in zip(transformed_probes, probes):
            if len(df) == 0:
                predictions.append(np.full(len(raw_probe), self.model.classes_[0]))
                continue

            proba = self.model.predict_proba(df)
            classes = self.model.classes_
            n_samples = len(raw_probe)
            increment = max(1, int(self.chunk_size * (1 - self.overlap)))

            votes = np.zeros((n_samples, len(classes)))
            counts = np.zeros(n_samples)
            starts = np.arange(0, n_samples - self.chunk_size + 1, increment)

            for i, start in enumerate(starts):
                end = min(start + self.chunk_size, n_samples)
                win_len = end - start
                if win_len <= 0:
                    continue
                sl = slice(start, end)
                votes[sl] += proba[i] * self.window_weight[:win_len, None]
                counts[sl] += self.window_weight[:win_len]

            last_start = max(0, n_samples - self.chunk_size)
            last_end = n_samples
            last_len = last_end - last_start
            if last_len > 0:
                last_sl = slice(last_start, last_end)
                last_weights = self.window_weight[self.chunk_size - last_len :]
                votes[last_sl] += proba[-1] * last_weights[:, None]
                counts[last_sl] += last_weights

            counts = np.clip(counts, 1e-12, None)
            avg_votes = votes / counts[:, None]
            predictions.append(classes[np.argmax(avg_votes, axis=1)])

        return predictions

    def save(self):
        with open(self.model_path, "ab") as f:
            pickle.dump(self.model, f)

    def load(self, path=None):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
