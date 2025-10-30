import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
import pickle
import optuna
import warnings
import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)

class Model():
    def __init__(self, save_path = None, trial = None):
        self.chunk_seconds = 3 #number of seconds - size of the window you look at.
        self.num_estimators = 128 #?
        self.num_freqs = 7 #?
        self.max_depth = 16
        self.sample_rate = 100 #?
        self.chunk_size = self.chunk_seconds * self.sample_rate #?
        self.waveform_type = "post_rect" #better than pre
        self.random_state = 42
        dirname = os.path.dirname(__file__)
        self.model = None
        self.save_path = save_path
        self.model_path = "../ML/rf_pickle"
        
        if trial: #?
            self.chunk_seconds = trial.suggest_int('chunk_seconds', 1, 3)
            self.num_freqs = trial.suggest_int('num_freqs', 1, 10)
            self.num_estimators = trial.suggest_categorical('num_estimators', [8, 16, 32, 64, 128])
            self.max_depth = trial.suggest_categorical('max_depth', [8, 16, 32, 64, 128])

    def transform_data(self, probes, training = True):
        transformed_probes = []
        for probe in probes:
            num_chunks = len(probe) // self.chunk_size
            if num_chunks == 0:
                print("num_chunks is 0!")
                print(len(probe))
                print(self.chunk_size)
            chunks = np.array_split(probe[:num_chunks * self.chunk_size], num_chunks) #so chunks is each probe split up into chunks of three seconds times 100 hz?
            columns = defaultdict(list)
            for chunk in chunks:
                chunk_fft = np.abs(fft(chunk[self.waveform_type].values))[1:self.chunk_size//2] #gets postrec values for each chunk, takes its abs value, 
                #The first element (index 0) of the FFT represents the DC component (zero frequency) - essentially the mean/average value of the signal
                #So we omit that with 1:
                chunk_freqs = fftfreq(self.chunk_size, 1 / self.sample_rate)[1:self.chunk_size//2]
                """
                The FFT of real-valued data is symmetric - the second half is a mirror image (complex conjugate) of the first half
With chunk_size = 300 samples (3 seconds × 100 Hz), you get 300 FFT values, but:
Indices 0 to 149 contain unique frequency information
Indices 150 to 299 are redundant (mirrored)
The Nyquist frequency is at chunk_size//2, which represents the maximum frequency you can detect (50 Hz in your case, which is half the 100 Hz sampling rate)
Everything beyond chunk_size//2 is redundant for real-valued signals
                """

                num_largest = self.num_freqs #7?
                indices = (-chunk_fft).argpartition(num_largest, axis=None)[:num_largest]
                #argpartition returns the smallest numbers in the arr (which if negative, returns largest)
                """It rearranges the indices so that the smallest k values are in the first k positions
The remaining indices go in positions k onward
It returns the entire rearranged array of indices"""
                indices = sorted(indices, key=lambda x: chunk_fft[x], reverse=True)

                peak_freqs = chunk_freqs[indices]

                for i in range(num_largest):
                    columns[f"F{i}"].append(peak_freqs[i])
                columns["mean"].append(np.mean(chunk[self.waveform_type]))
                columns["std"].append(np.std(chunk[self.waveform_type]))
                columns["resistance"].append(chunk["resistance"].values[0])
                columns["volts"].append(chunk["voltage"].values[0])
                columns["current"].append(0 if chunk["current"].values[0] == "AC" else 1)
                if training: # In reality, we won't know what the labels are
                    labels, label_counts = np.unique(chunk["labels"], return_counts=True)
                    label = labels[np.argmax(label_counts)]
                    columns["label"].append(label)

            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)
        return transformed_probes

    def train(self, probes, test_data, fold):
        transformed_probes = self.transform_data(probes)
        train = pd.concat(transformed_probes)
        X_train = train.drop(["label"], axis=1)
        Y_train = train["label"]
        rf = RandomForestClassifier(self.num_estimators, class_weight="balanced", max_depth = self.max_depth)
        self.model = rf.fit(X_train, Y_train)
    
    def predict(self, probes):
        transformed_probes = self.transform_data(probes, training = False)
        predictions = []
        for transformed_probe, raw_probe in zip(transformed_probes, probes):
            test_probe = transformed_probe
            pred = self.model.predict(test_probe)

            # We need to expand the prediction based on the sample rate
            pred = np.repeat(pred, self.chunk_seconds * self.sample_rate)
            # Expand until the end since probe is never exactly divisible by window size
            pred = np.pad(pred, (0, len(raw_probe) - len(pred)), 'edge')
            predictions.append(pred)
        return predictions

    def save(self):
        with open(self.model_path, 'ab') as model_save:
            pickle.dump(self.model, model_save)

    def load(self, path = None):
        with open(path, 'rb') as model_save:
            self.model = pickle.load(model_save)
