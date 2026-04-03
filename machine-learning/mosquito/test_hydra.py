import os
import glob
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            accuracy_score, \
                            f1_score
from matplotlib import pyplot as plt
from itertools import groupby
import optuna
from sklearn.model_selection import train_test_split
from utilities.model_factory import build_model 
from utilities.train import run_experiment
from data_augmentation import build_augmented_dataset
from postprocessing import PostProcessor
import hydra
from omegaconf import DictConfig


class DataImport:
        def __init__(self, data_path, folds):
            self.raw_dfs, _ = self.import_data(data_path)
            self.random_state = 57 # 42
            kf = KFold(n_splits = folds, random_state = self.random_state,\
                       shuffle = True)
            self.cross_val_iter = list(kf.split(self.raw_dfs))
            self.probe_finder_method = self.leak_probe_finder

        def get_train_test_split(self, test_size=0.2):
            train_dfs, test_dfs = train_test_split(self.raw_dfs, test_size = test_size,
                                                             random_state = 42)
            
            train_probes, _ = self.get_probes(train_dfs)
            test_probes, test_names = self.get_probes(test_dfs)
            return train_probes, test_probes, test_names

        def import_data(self, data_path):
            """
            import_data takes in a path to cleaned data and returns it
            as a list of dataframes.
            """
            filenames = glob.glob(os.path.expanduser(f"{data_path}/*.csv"))
            dataframes = [pd.read_csv(f) for f in filenames]
            for file, df in zip(filenames, dataframes):
                df["file"] = file
                df['labels'] = df['labels'].replace('Z', 'W')
            return dataframes, filenames

        def leak_probe_finder(self, labels):
            non_np_indices = np.where(labels != 'NP')[0]
            probes = []
            start = 0
            end = 0
            for i in range(len(non_np_indices)):
                if i == 0:
                    start = non_np_indices[i]
                elif i == len(non_np_indices) - 1:
                    end = non_np_indices[i]
                elif abs(non_np_indices[i] - non_np_indices[i - 1]) > 1:
                    start = non_np_indices[i]
                elif abs(non_np_indices[i] - non_np_indices[i + 1]) > 1:
                    end = non_np_indices[i]
                if start > 0 and end > 0:
                    probes.append((start, end))
                    start = 0
                    end = 0
            return probes

        def simple_probe_finder(self, recording, window = 500, threshold = 0.1,
                         min_probe_length = 1500, np_pad = 500):
            """
            Input: recording: A pre-rectified mosquito recording as an 1-D nupmy 
                     array. Pre-rectified recordings are necessary as baseline is 
                     not 0 in post-rectified recordings.
                   window: Before NP regions can be identified, a rolling
                     average filter is applied to remove noise in the NP regions.
                     window is the size of this filter in samples.
                   threshold: The maximum value of an NP sample.
                   min_probe_length: The minimum acceptable length of a probe in
                     samples.
                   np_pad: the number of NP samples before and after each probe to
                     include. Note that high values might result in overlapping with
                     the next probe.
            Output: A list of (start sample, end sample) tuples for each probe. By 
                    default contains about 5 seconds of NPs at the beginning and end            
                    of each probe. We say "about" because this splitting is done
                    in an unsupervised manner, although it is largely pretty good.
            """
            
            smoothed = np.convolve(recording, np.ones(window), "same")/window
            is_NP = smoothed < threshold # NP is where the signal is close to 0
            
            # Find starts and ends, combine into tuple
            find_sequence = lambda l, seq : [i for i in range(len(l)) if l[i:i+len(seq)] == seq]
            is_NP_list = list(is_NP)
            probe_starts = find_sequence(is_NP_list, [True, False])
            probe_ends = find_sequence(is_NP_list, [False, True])
            probes = zip(probe_starts, probe_ends)
            
            # Remove probes that are too short and pad
            probes = [(max(0, start - np_pad), end + np_pad) for start, end in probes if end - start > min_probe_length]
            
            return probes

        def get_probes(self, dfs):
            """
            Input: probe_finder_method: the method by which to find probes
            Output: a list of dataframes, each consisting of a probe along with a list of names
            """
            all_probes = []
            all_probe_names = []
            for df in dfs:
                probe_indices = self.probe_finder_method(df["labels"].values)
                probes = [df.iloc[start:end].reset_index(drop=True).copy() 
                          for start, end in probe_indices]
                probe_names = [df["file"][0][:-4] + f"_{str(i)}" 
                               for i, df in enumerate(probes)]
                
                all_probes.extend(probes)
                all_probe_names.extend(probe_names)

            return all_probes, all_probe_names
        
@hydra.main(version_base=None, config_path="conf", config_name="config")        
def main(cfg: DictConfig):
    print(cfg)
    model = build_model(cfg.model)
    print(model.model)

if __name__ == "__main__":
    main()