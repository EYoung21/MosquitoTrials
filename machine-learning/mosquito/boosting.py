from typing import Any


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import optuna
import warnings
import tqdm
from tabular_model import TabularModel
warnings.simplefilter(action='ignore', category=FutureWarning)

class Model(TabularModel):
    def __init__(self, save_path = None, trial = None):
        super().__init__()  # Initialize base class
        
        #chunk hyperparameters
        self.chunk_seconds = 1
        self.num_estimators = 100  # Number of boosting stages
        self.learning_rate = 0.1  # Learning rate shrinks the contribution of each tree
        self.num_freqs = 7  # Number of largest frequencies in each window to extract as a feature
        self.sample_rate = 100
        self.chunk_size = int(self.chunk_seconds * self.sample_rate)  # Convert to int for array indexing
        self.window_seconds = 3
        self.window_size = self.window_seconds * self.sample_rate

        self.max_depth = 3  # Maximum depth of the individual regression estimators (shallow for boosting)
        self.min_samples_split = 2  # Minimum number of samples required to split an internal node
        self.min_samples_leaf = 1  # Minimum number of samples required to be at a leaf node
        self.subsample = 1.0  # Fraction of samples to be used for fitting the individual base learners
        self.waveform_type = "post_rect"
        self.random_state = 42
        dirname = os.path.dirname(__file__)
        self.model = None
        self.save_path = save_path
        self.model_path = "../ML/boosting_pickle"
        
        if trial:
            self.chunk_seconds = trial.suggest_float('chunk_seconds', 0.01, 2.0)
            self.num_freqs = trial.suggest_int('num_freqs', 1, 20)
            self.num_estimators = trial.suggest_int('num_estimators', 50, 500)
            self.learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            self.max_depth = trial.suggest_int('max_depth', 3, 8)
            self.min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            self.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            self.subsample = trial.suggest_float('subsample', 0.6, 1.0)
            self.window_seconds = trial.suggest_int('window_seconds', 2, 5)
            
            # Recalculate derived values based on trial suggestions
            self.chunk_size = int(self.chunk_seconds * self.sample_rate)  # Convert to int for array indexing
            self.window_size = self.window_seconds * self.sample_rate

    def train(self, probes, test_data, fold):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        
        # Initialize Gradient Boosting Classifier with optimized hyperparameters
        gb = GradientBoostingClassifier(
            n_estimators=self.num_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
            verbose=0
        )
        self.model = gb.fit(X_train, Y_train)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training = False) #transformed_probes is just each probe (chunk?) with al the features attached
        predictions = []
        for transformed_probe, raw_probe in zip(transformed_probes, probes):
            test_probe = transformed_probe
            pred = self.model.predict(test_probe)

            # we need to expand the prediction based on the sample rate
            pred = np.repeat(pred, int(self.chunk_seconds * self.sample_rate)) #what does this do?!
            # expand until the end since probe is never exactly divisible by window size
            pred = np.pad(pred, (0, len(raw_probe) - len(pred)), 'edge') #would this alter our prediction?!
            predictions.append(pred)
        return predictions

    def save(self):
        with open(self.model_path, 'ab') as model_save:
            pickle.dump(self.model, model_save)

    def load(self, path = None):
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)
