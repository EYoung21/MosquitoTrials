from rf_sunwindow_smote import Model as BaseModel


class Model(BaseModel):
    """
    Same as rf_sunwindow_smote, but lowers chunk_seconds search floor
    from 0.1s to 0.05s to test whether the previous best-on-bound persists.
    """

    def __init__(self, save_path=None, trial=None):
        super().__init__(save_path=save_path, trial=None)

        if trial:
            self.window_seconds = trial.suggest_int("window_seconds", 2, 40)
            self.window_size = self.window_seconds * self.sample_rate
            max_chunk_sec = float(self.window_seconds)
            self.chunk_seconds = trial.suggest_float("chunk_seconds", 0.05, max_chunk_sec, step=0.05)
            self.chunk_size = max(1, int(round(self.chunk_seconds * self.sample_rate)))
            self.num_freqs = trial.suggest_int("num_freqs", 1, 30)
            self.num_estimators = trial.suggest_categorical("num_estimators", [8, 16, 32, 64, 128, 256, 512])
            self.max_depth = trial.suggest_categorical("max_depth", [4, 8, 16, 32, 64, 128, 256, 512])
            self.num_subwindows = trial.suggest_int("num_subwindows", 1, 30)
            self.subwindow_size = int(self.window_size / self.num_subwindows)
            self.subwindow_freq = trial.suggest_int("subwindow_freq", 1, 30)
            self.use_window_slope = trial.suggest_categorical("use_window_slope", [True, False])
            self.use_subwindow_slope = trial.suggest_categorical("use_subwindow_slope", [True, False])
            self.use_smote = trial.suggest_categorical("use_smote", [True, False])
            if self.use_smote:
                self.smote_k_neighbors = trial.suggest_int("smote_k_neighbors", 1, 50)
                self.smote_sampling_strategy = "auto"
