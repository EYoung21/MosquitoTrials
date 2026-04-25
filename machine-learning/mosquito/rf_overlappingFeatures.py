from typing import Any


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
        self.chunk_seconds = 1 #trying out second window (with three seconds (1 left, 1 right) for input feature isolatopm)
         #the chunk size, number of seconds - size of the window you look at. 100hz*3 = 300hz
        self.num_estimators = 64 # OPTIMIZED: was 128
        self.num_freqs = 17 # OPTIMIZED: was 7
        self.sample_rate = 100 #?
        self.chunk_size = self.chunk_seconds * self.sample_rate
        # self.windowMultiplier = 3
        self.window_seconds = 5 # OPTIMIZED: was 3
        self.window_size = self.window_seconds * self.sample_rate #multiplying chunk size by three here to incooperate overlapping features

        self.max_depth = 16 #where to stop splitting
        self.waveform_type = "post_rect" #better than pre
        self.random_state = 42
        dirname = os.path.dirname(__file__)
        self.model = None
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"
        
        if trial: #?
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 10)
            self.num_freqs = trial.suggest_int('num_freqs', 1, 20)
            self.num_estimators = trial.suggest_categorical('num_estimators', [8, 16, 32, 64, 128, 256])
            self.max_depth = trial.suggest_categorical('max_depth', [4, 8, 16, 32, 64, 128, 256])
            self.window_seconds = trial.suggest_int('window_seconds', 2, 10)
            
            # Recalculate derived values based on trial suggestions
            self.chunk_size = self.chunk_seconds * self.sample_rate
            self.window_size = self.window_seconds * self.sample_rate

    def train(self, probes, test_data, fold):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        rf = RandomForestClassifier(self.num_estimators, class_weight="balanced", max_depth = self.max_depth)
        self.model = rf.fit(X_train, Y_train)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training = False) #transformed_probes is just each probe (chunk?) with al the features attached
        predictions = []
        for transformed_probe, raw_probe in zip(transformed_probes, probes):
            # Handle probes that were too short and skipped during transform_data
            if len(transformed_probe) == 0:
                # Return default prediction (label 0) for the entire probe
                pred = np.zeros(len(raw_probe), dtype=int)
                predictions.append(pred)
                continue
            
            test_probe = transformed_probe
            pred = self.model.predict(test_probe)

            # we need to expand the prediction based on the sample rate
            pred = np.repeat(pred, self.chunk_seconds * self.sample_rate) #what does this do?!
            # expand until the end since probe is never exactly divisible by window size
            pad_length = len(raw_probe) - len(pred)
            if pad_length > 0:
                pred = np.pad(pred, (0, pad_length), 'edge') #would this alter our prediction?!
            elif pad_length < 0:
                # Truncate if predictions are somehow longer
                pred = pred[:len(raw_probe)]
            predictions.append(pred)
        return predictions

    def save(self):
        with open(self.model_path, 'ab') as model_save:
            pickle.dump(self.model, model_save)

    def load(self, path = None):
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)
