from typing import Any


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pickle
import optuna
import warnings
import tqdm
warnings.simplefilter(action='ignore', category=FutureWarning)

class Model():
    def __init__(self, save_path = None, trial = None):
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

    def transform_data(self, probes, training = True):
        transformed_probes = []
        # chunksToWindow = dict[Any, Any]()
        #maybe dont need dictionary actually
        for probe in probes:
            num_chunks = len(probe) // self.chunk_size
            if num_chunks == 0:
                print(f"Skipping probe: too short (len={len(probe)}, chunk_size={self.chunk_size})")
                continue  # Skip probes that are too short
            chunks = np.array_split(probe[:num_chunks * self.chunk_size], num_chunks) #for each probe, split it up into chunks of a predefined number of seconds times 100 hz

            columns = defaultdict(list)
            for i, chunk in enumerate(chunks):
                extra_context_size = (self.window_size - self.chunk_size)//2

                chunkStartIndex = i * self.chunk_size
                window_start = max(0, chunkStartIndex - extra_context_size)
                window_end = min(len(probe), chunkStartIndex + self.chunk_size + extra_context_size)
                currWindow = probe[window_start:window_end]

                chunk_fft = np.abs(fft(currWindow[self.waveform_type].values))[1:self.window_size//2]  #changed to calculate featur on currWinodw
                #fourier transform, gets largest frequencies. gets postrec values for each chunk, takes its abs value.
                #the first element (index 0) of the FFT represents the DC component (zero frequency) - essentially the mean/average value of the signal
                #so we omit that with 1:
                chunk_freqs = fftfreq(self.window_size, 1 / self.sample_rate)[1:self.window_size//2] #gets the size of freq for each chunk
                #skip index 0 (the DC component/zero frequency)
                #take only the positive frequencies up to the Nyquist frequency 
                #(half the chunk size) (skips part of fourier transform that is reversed, meaningless.)
                """
                the FFT of real-valued data is symmetric - the second half is a mirror image (complex conjugate) of the first half
                with chunk_size = 300 samples (3 seconds × 100 Hz), you get 300 FFT values, but:
                indices 0 to 149 contain unique frequency information
                indices 150 to 299 are redundant (mirrored)
                the Nyquist frequency is at chunk_size//2, which represents the maximum frequency you can detect (50 Hz in your case, which is half the 100 Hz sampling rate)
                everything beyond chunk_size//2 is redundant for real-valued signals
                """

                
                num_largest = self.num_freqs #7
                indices = (-chunk_fft).argpartition(num_largest, axis=None)[:num_largest]
                #get the largest frequencies (or the smallest negative ones)
                #argpartition returns the smallest numbers in the arr (which if negative, returns largest)
                """it rearranges the indices so that the smallest k values are in the first k positions
                the remaining indices go in positions k onward
                it returns the entire rearranged array of indices"""
                indices = sorted(indices, key=lambda x: chunk_fft[x], reverse=True)
                #sorts indices by frequency size

                peak_freqs = chunk_freqs[indices] #gets the actual values of the largest 7 frequences.

                for i in range(num_largest):
                    columns[f"F{i}"].append(peak_freqs[i]) #7, or x, largest frequences in chunk
                columns["mean"].append(np.mean(currWindow[self.waveform_type])) #mean postrec
                columns["std"].append(np.std(currWindow[self.waveform_type]))#std of postrec
                
                # Fit linear regression to capture signal trend
                window_signal = currWindow[self.waveform_type].values
                time_steps = np.arange(len(window_signal)).reshape(-1, 1)  # Time as feature
                lr = LinearRegression()
                lr.fit(time_steps, window_signal)
                columns["trend_coef"].append(lr.coef_[0])  # Slope of the trend
                columns["trend_intercept"].append(lr.intercept_)  # Intercept of the trend
                
                columns["resistance"].append(currWindow["resistance"].values[0]) #?, why [0]?
                columns["volts"].append(currWindow["voltage"].values[0]) #??, why [0]?s
                columns["current"].append(0 if currWindow["current"].values[0] == "AC" else 1) #AC (?) or not, binary?
                if training: # in reality, we won't know what the labels are
                    labels, label_counts = np.unique(currWindow["labels"], return_counts=True) #probing labels
                    label = labels[np.argmax(label_counts)]
                    columns["label"].append(label)

            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)#what is this?
        return transformed_probes

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
