"""
Base class for tabular EPG models.

This class provides shared data preprocessing functionality that transforms
raw EPG probe data into feature vectors for machine learning models.
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from collections import defaultdict
from sklearn.linear_model import LinearRegression


class TabularModel:
    """
    Abstract base class for tabular models that process EPG data.
    
    This class handles the common data preprocessing pipeline including:
    - Chunking probes into time windows
    - FFT feature extraction
    - Statistical feature computation (mean, std, trend)
    - Hardware setting features (resistance, voltage, current)
    """
    
    def __init__(self):
        """
        Initialize base model parameters.
        
        Subclasses should call super().__init__() and set their own
        model-specific hyperparameters.
        """
        # Default hyperparameters - subclasses should override these
        self.chunk_seconds = 1
        self.num_freqs = 7
        self.sample_rate = 100
        self.chunk_size = self.chunk_seconds * self.sample_rate
        self.window_seconds = 3
        self.window_size = self.window_seconds * self.sample_rate
        self.waveform_type = "post_rect"
        self.random_state = 42
        self.model = None
        self.save_path = None
        
    def transform_data(self, probes, training=True):
        """
        Transform raw probe data into feature vectors for ML models.
        
        This method:
        1. Splits each probe into chunks based on chunk_seconds
        2. For each chunk, extracts features from a sliding window context
        3. Computes FFT features (dominant frequencies)
        4. Computes statistical features (mean, std, trend)
        5. Includes hardware settings (resistance, voltage, current)
        6. Optionally includes labels for training
        
        Args:
            probes: List of DataFrames, each containing raw EPG data for one probe
            training: If True, includes labels in the output
            
        Returns:
            List of DataFrames, each containing feature vectors for one probe
        """
        transformed_probes = []
        
        for probe in probes:
            num_chunks = len(probe) // self.chunk_size
            if num_chunks == 0:
                print(f"Skipping probe: too short (len={len(probe)}, chunk_size={self.chunk_size})")
                continue  # Skip probes that are too short
                
            # Split probe into chunks
            chunks = np.array_split(probe[:num_chunks * self.chunk_size], num_chunks)
            
            columns = defaultdict(list)
            
            for i, chunk in enumerate(chunks):
                # Calculate window with extra context around the chunk
                extra_context_size = (self.window_size - self.chunk_size) // 2
                chunk_start_index = i * self.chunk_size
                window_start = max(0, chunk_start_index - extra_context_size)
                window_end = min(len(probe), chunk_start_index + self.chunk_size + extra_context_size)
                curr_window = probe[window_start:window_end]
                
                # FFT features: Extract dominant frequencies
                chunk_fft = np.abs(fft(curr_window[self.waveform_type].values))[1:self.window_size//2]
                chunk_freqs = fftfreq(self.window_size, 1 / self.sample_rate)[1:self.window_size//2]
                
                # Find the largest frequency components
                num_largest = self.num_freqs
                indices = (-chunk_fft).argpartition(num_largest, axis=None)[:num_largest]
                indices = sorted(indices, key=lambda x: chunk_fft[x], reverse=True)
                peak_freqs = chunk_freqs[indices]
                
                # Add frequency features
                for j in range(num_largest):
                    columns[f"F{j}"].append(peak_freqs[j])
                
                # Statistical features
                columns["mean"].append(np.mean(curr_window[self.waveform_type]))
                columns["std"].append(np.std(curr_window[self.waveform_type]))
                
                # Trend features: Fit linear regression to capture signal trend
                window_signal = curr_window[self.waveform_type].values
                time_steps = np.arange(len(window_signal)).reshape(-1, 1)
                lr = LinearRegression()
                lr.fit(time_steps, window_signal)
                columns["trend_coef"].append(lr.coef_[0])  # Slope of the trend
                columns["trend_intercept"].append(lr.intercept_)  # Intercept of the trend
                
                # Hardware setting features
                columns["resistance"].append(curr_window["resistance"].values[0])
                columns["volts"].append(curr_window["voltage"].values[0])
                columns["current"].append(0 if curr_window["current"].values[0] == "AC" else 1)
                
                # Labels (for training only)
                if training:
                    labels, label_counts = np.unique(curr_window["labels"], return_counts=True)
                    label = labels[np.argmax(label_counts)]
                    columns["label"].append(label)
            
            probe_out = pd.DataFrame(columns)
            transformed_probes.append(probe_out)
            
        return transformed_probes
    
    def train(self, probes, test_data, fold):
        """
        Train the model on probe data.
        
        Subclasses must implement this method to define their training logic.
        
        Args:
            probes: Training probe data
            test_data: Test/validation data
            fold: Current fold number for cross-validation
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, probes):
        """
        Make predictions on probe data.
        
        Subclasses must implement this method to define their prediction logic.
        
        Args:
            probes: List of probes to make predictions on
            
        Returns:
            List of prediction arrays, one per probe
        """
        raise NotImplementedError("Subclasses must implement predict()")
    
    def save(self):
        """
        Save the trained model.
        
        Subclasses should implement this if they need custom save logic.
        """
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self, path=None):
        """
        Load a trained model.
        
        Subclasses should implement this if they need custom load logic.
        """
        raise NotImplementedError("Subclasses must implement load()")
