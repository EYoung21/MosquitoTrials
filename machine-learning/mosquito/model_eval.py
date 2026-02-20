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
import importlib.util
from matplotlib import pyplot as plt
from itertools import groupby
import optuna
from sklearn.model_selection import train_test_split
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from data_augmentation import build_augmented_dataset
from postprocessing import PostProcessor

class DataImport:
        def __init__(self, data_path, folds):
            self.raw_dfs, _ = self.import_data(data_path)
            self.random_state = 57 # 42
            kf = KFold(n_splits = folds, random_state = self.random_state,\
                       shuffle = True)
            self.cross_val_iter = list(kf.split(self.raw_dfs))
            self.probe_finder_method = self.leak_probe_finder

        def get_train_test_split(self):
            train_dfs, test_dfs = train_test_split(self.raw_dfs, test_size = 0.2,
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

def dynamic_importer(path):
    spec = importlib.util.spec_from_file_location("model", path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    return model

def plot_labels(time, voltage, true_labels, pred_labels, probs = None):
    """
    plot_labels produced a matplotlib figure containing three subplots
        that visualize a waveform along with the true and predicted labels
    Input:
        time: a series of time values
        voltage: a time series of voltage values from the waveform
        true_labels: a time series of the true label for each time point
        pred_labels: a time series of the predicted labels for each time point
    Output:
        (fig, axs): a tuple
    """
    label_to_color = {
            "NP": "red",
            "J": "blue",
            "K": "green",
            "L": "purple",
            "M": "pink",
            "N": "cyan",
            "W": "orange"
    }

    fig, axs = plt.subplots(3, 1, sharex = True)
    recording = 1
    fill_min, fill_max = voltage.min(), voltage.max()
    
    # First plot will be the true labels
    axs[0].plot(time, voltage, color = "black")
    for label, color in label_to_color.items():
        fill = axs[0].fill_between(time, fill_min, fill_max, 
                where = (true_labels == label), color=color, alpha = 0.5)
        fill.set_label(label)
    axs[0].legend(bbox_to_anchor=(0.5, 1), 
                  bbox_transform=fig.transFigure, loc="upper center", ncol=9)
    axs[0].set_title("True Labels")
    # Second plot will be the predicted labels
    axs[1].plot(time, voltage, color = "black")
    for label, color in label_to_color.items():
        axs[1].fill_between(time, fill_min, fill_max, 
                where = (pred_labels == label), color=color, alpha = 0.5)
    axs[1].set_title("Predicted Labels")
    # Third plot will be marked where there is a difference between the two
    axs[2].plot(time, voltage, color = "black")
    axs[2].fill_between(time, fill_min, fill_max, 
            where = (pred_labels != true_labels), color = "gray", alpha = 0.5)
    axs[2].set_title("Incorrect Labels")
    # Axes titles and such
    fig.supxlabel("Time (s)")
    fig.supylabel("Volts")
    fig.tight_layout()
    return fig



def generate_report(test_data, predicted_labels, test_names, save_path, model_name, fold, predicted_logits = None, inv_label_map = None):
    # Flatten everything
    labels_true = []
    labels_pred = []
    for df, preds in zip(test_data, predicted_labels):
        labels_true.extend(df["labels"].values)
        labels_pred.extend(preds)

    # Make sure we have a place to save everything
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # precision et. al
    labels = sorted(np.unique(labels_true))
    precision, recall, fscore, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = None, zero_division=0)
    temp_dict = {"precision" : precision, 
                 "recall" : recall, 
                 "fscore" : fscore}
    out_dataframe = pd.DataFrame(temp_dict, index=labels).stack()
    out_dataframe.index = out_dataframe.index.map('{0[1]}_{0[0]}'.format)
    out_dataframe = out_dataframe.to_frame().T

    # accuracy
    accuracy = accuracy_score(labels_true, labels_pred)
    out_dataframe["accuracy"] = accuracy

    all_precision_micro, all_recall_micro, all_fscore_micro, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = "micro", zero_division=0)
    all_precision_macro, all_recall_macro, all_fscore_macro, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = "macro", zero_division=0)
    out_dataframe["precision_micro"] = all_precision_micro
    out_dataframe["recall_micro"] = all_recall_micro
    out_dataframe["fscore_micro"] = all_fscore_micro
    out_dataframe["precision_macro"] = all_precision_macro
    out_dataframe["recall_macro"] = all_recall_macro
    out_dataframe["fscore_macro"] = all_fscore_macro

    # confusion matrix
    ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, \
                                            normalize = 'true')
    plt.savefig(rf"{save_path}/{model_name}_ConfusionMatrix_Fold{fold}.png")

    # difference plots
    for i, (df, preds, name) in enumerate(zip(test_data, predicted_labels, test_names)):
        fig = plot_labels(df["time"], df["pre_rect"], df["labels"].values, np.array(preds))
        fig.savefig(fr"{save_path}/{model_name}_{os.path.split(name)[1]}_Fold{fold}.png")
        plt.close(fig)

    print(f"Fold {fold} Overall Accuracy: {accuracy}")
    return labels_true, labels_pred, out_dataframe

def generate_roc(
    test_data,
    all_logits,
    save_path=None,
    model_name="model",
    fold=0,
    labels=("J","K","L","M","N","W"),
    class_colors=None,
    score_transform="identity",  # "identity", "sigmoid", or "softmax"
    mark_argmax_operating_point=False,
    point_size=80,
):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    # Color map
    default_colors = {
        "J": "blue", "K": "green", "L": "purple",
        "M": "pink", "N": "cyan", "W": "orange"
    }
    label_to_color = default_colors if class_colors is None else class_colors

    # ---- collect y and scores ----
    y_true_labels = []
    score_frames = []
    for df, logits in zip(test_data, all_logits):
        y_true_labels.extend(df["labels"].astype(str).values)
        score_frames.append(pd.DataFrame(logits, columns=labels))
    scores = pd.concat(score_frames, ignore_index=True).astype(float)  # shape [N, C]

    # ---- score transforms ----
    if score_transform == "sigmoid":
        x = scores.values
        scores = pd.DataFrame(1.0 / (1.0 + np.exp(-x)), columns=labels)
    elif score_transform == "softmax":
        x = scores.values
        x = x - x.max(axis=1, keepdims=True)         # numerical stability
        ex = np.exp(x)
        scores = pd.DataFrame(ex / ex.sum(axis=1, keepdims=True), columns=labels)
    # else: identity (raw logits)

    # ---- binarize y (OvR) ----
    y_test = label_binarize(y_true_labels, classes=list(labels))  # shape [N, C]
    y_score = scores.values
    n_classes = y_test.shape[1]
    N = y_score.shape[0]

    # ---- per-class curves ----
    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, pr_auc = {}, {}, {}

    valid_class_idx = []
    for i in range(n_classes):
        # need at least one positive and one negative to compute curve
        has_pos = (y_test[:, i].sum() > 0)
        has_neg = ((1 - y_test[:, i]).sum() > 0)
        if not (has_pos and has_neg):
            continue
        valid_class_idx.append(i)

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        pr_auc[i] = auc(recall[i], precision[i])

    # ---- micro averages ----
    if len(valid_class_idx) > 0:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test[:, valid_class_idx].ravel(),
                                                  y_score[:, valid_class_idx].ravel())
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test[:, valid_class_idx].ravel(), y_score[:, valid_class_idx].ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        pr_auc["micro"] = auc(recall["micro"], precision["micro"])

        # ---- macro averages (simple interpolation) ----
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_class_idx]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in valid_class_idx:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid_class_idx)
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        all_recall = np.unique(np.concatenate([recall[i] for i in valid_class_idx]))
        mean_precision = np.zeros_like(all_recall)
        for i in valid_class_idx:
            mean_precision += np.interp(all_recall, recall[i], precision[i])
        mean_precision /= len(valid_class_idx)
        recall["macro"], precision["macro"] = all_recall, mean_precision
        pr_auc["macro"] = auc(recall["macro"], precision["macro"])

    # ---- default argmax operating point (per class) ----
    # This is the discrete classifier that predicts argmax across classes.
    # For each class i, compute its OvR confusion stats under argmax.
    argmax_idx = y_score.argmax(axis=1)  # predicted class per sample
    op_points = {}  # i -> dict with fpr,tpr,prec,recall,f1
    for i in range(n_classes):
        y_true_i = (y_test[:, i] == 1)
        y_pred_i = (argmax_idx == i)

        tp = np.sum(y_pred_i & y_true_i)
        fp = np.sum(y_pred_i & (~y_true_i))
        fn = np.sum((~y_pred_i) & y_true_i)
        tn = N - tp - fp - fn

        # metrics with safe division
        tpr_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_i = tpr_i
        f1_i = (2 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0
        op_points[i] = dict(fpr=fpr_i, tpr=tpr_i, prec=prec_i, rec=rec_i, f1=f1_i)

    # ---- ensure output dir ----
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    lw = 3

    # ===================== ROC plot =====================
    plt.figure()
    if "micro" in fpr:
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f"micro-average ROC (AUC = {roc_auc['micro']:.2f})",
                 linestyle=":", linewidth=lw, color="deeppink")
    if "macro" in fpr:
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f"macro-average ROC (AUC = {roc_auc['macro']:.2f})",
                 linestyle=":", linewidth=lw, color="navy")

    for i, lab in enumerate(labels):
        if i in valid_class_idx:
            plt.plot(fpr[i], tpr[i], label=f"{lab} (AUC = {roc_auc[i]:.2f})",
                     lw=lw, color=label_to_color.get(lab, None))

        if mark_argmax_operating_point:
            pt = op_points[i]
            plt.scatter(pt["fpr"], pt["tpr"], s=point_size,
                        edgecolor="k", linewidths=0.7,
                        color=label_to_color.get(lab, None), alpha=0.9,
                        label=None)
            # Annotate with F1 near the point
            plt.annotate(f"F1={pt['f1']:.2f}", (pt["fpr"], pt["tpr"]),
                         xytext=(5, -10), textcoords="offset points", fontsize=8)

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim(0, 1); plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend(loc="lower right", ncol=2, fontsize=8)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{model_name}_ROC_Fold{fold}.png"), dpi=200)
        plt.close()
    else:
        plt.show()

    # ===================== PR plot =====================
    plt.figure()
    if "micro" in precision:
        plt.plot(recall["micro"], precision["micro"],
                 label=f"Average PR (AUC = {pr_auc['micro']:.2f})",
                 linestyle=":", linewidth=lw, color="deeppink")
    # if "macro" in precision:
    #     plt.plot(recall["macro"], precision["macro"],
    #              label=f"macro-average PR (AUC = {pr_auc['macro']:.2f})",
    #              linestyle=":", linewidth=lw, color="navy")

    for i, lab in enumerate(labels):
        if i in valid_class_idx:
            plt.plot(recall[i], precision[i], label=f"{lab} (AUC = {pr_auc[i]:.2f})",
                     lw=lw, color=label_to_color.get(lab, None))

        if mark_argmax_operating_point:
            pt = op_points[i]
            plt.scatter(pt["rec"], pt["prec"], s=point_size,
                        edgecolor="k", linewidths=0.7,
                        color=label_to_color.get(lab, None), alpha=0.9,
                        label=None)
            plt.annotate(f"F1={pt['f1']:.2f}", (pt["rec"], pt["prec"]),
                         xytext=(5, -10), textcoords="offset points", fontsize=8)

    plt.xlim(0, 1); plt.ylim(0, 1.05)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall curves")
    plt.legend(loc="lower left", ncol=2, fontsize=8)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{model_name}_PR_Fold{fold}.png"), dpi=200)
        plt.close()
    else:
        plt.show()

def apply_best_params_to_kwargs(best_params: dict, kwargs: dict) -> dict:
    """
    Overwrite model kwargs with Optuna best_params, with a small amount of
    key-aliasing to match your non-optuna kwargs naming.
    """
    new_kwargs = dict(kwargs)

    # direct overwrite when keys already exist in kwargs
    for k, v in best_params.items():
        if k in new_kwargs:
            new_kwargs[k] = v

    # common aliases between optuna params and your kwargs naming
    if "dropout" in best_params and "dropout_rate" in new_kwargs:
        new_kwargs["dropout_rate"] = best_params["dropout"]
    if "dropout_rate" in best_params and "dropout" in new_kwargs:
        new_kwargs["dropout"] = best_params["dropout_rate"]

    return new_kwargs

def optuna_objective(data, args, trial, outer_train_index, outer_fold, inner_folds=5, **kwargs):
    """
    Nested CV objective:
      - outer_train_index is the *outer* fold train split (indices into data.raw_dfs)
      - runs an inner KFold over outer_train_index
      - returns macro-F1 aggregated across inner validation folds
    """
    
    # Initialize a nested run for this trial
    run = wandb.init(
        project="hmc-epg-mosquito",
        group=f"{args.model_name}_optuna_outer{outer_fold}",
        name=f"trial_{trial.number}",
        config=trial.params,
        reinit=True,
        tags=["optuna", "nested"]
    )
    
    labels_true = []
    labels_pred = []

    augment_factor = 1 #trial.suggest_categorical("augment_factor", [1, 2, 4, 8])

    # Inner CV over the outer-train set only
    inner_kf = KFold(
        n_splits=inner_folds,
        random_state=data.random_state + outer_fold,  # deterministic per outer fold
        shuffle=True,
    )

    outer_train_index = np.array(list(outer_train_index))
    model_import = dynamic_importer(args.model_path)

    for inner_fold, (inner_train_pos, inner_val_pos) in enumerate(inner_kf.split(outer_train_index)):
        inner_train_idx = outer_train_index[inner_train_pos]
        inner_val_idx   = outer_train_index[inner_val_pos]

        train_dfs = [data.raw_dfs[i] for i in inner_train_idx]
        val_dfs   = [data.raw_dfs[i] for i in inner_val_idx]

        train_data, _ = data.get_probes(train_dfs)
        val_data, _   = data.get_probes(val_dfs)

        if args.augment:
            train_data = build_augmented_dataset(train_data, size=len(train_data) * augment_factor)

        model = model_import.Model(trial=trial, **kwargs)
        model.train(train_data, val_data, inner_fold)

        predicted_labels = model.predict(val_data)

        # Flatten everything (same as your original objective)
        for df, preds in zip(val_data, predicted_labels):
            labels_true.extend(df["labels"].values)
            labels_pred.extend(preds)
        break

    f1 = f1_score(labels_true, labels_pred, average="macro")
    print(f1_score(labels_true, labels_pred, average=None))

    with open(f"{args.model_name}_optuna.txt", "a") as f:
        print("outer_fold", outer_fold, trial.datetime_start, trial.number, trial.params, f1, file=f)

    if run is not None:
        run.log({"trial.macro_f1": f1})
        run.finish()

    return f1


def main():
    parser = argparse.ArgumentParser(
        prog = "Model Performance Evaluator",
        description = "This program takes in EPG data and a \
                        labeler program, trains it, and then \
                        generates statistics and figures to \
                        characterize the model's performance."
    )
    parser.add_argument("--data_path", type = str, required = True)
    parser.add_argument("--model_path", type = str, required = True)
    parser.add_argument("--save_path", type = str, required = True)
    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--post_process", type = str, required = False) # can either be s/smooth or viterbi/m
    parser.add_argument("--epochs", type = int, required=False)
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--attention", action="store_true") # can only be used with UNet
    parser.add_argument("--fold", type = int, required = False, default = -1) 
    args = parser.parse_args()

    print("Loading Data...")
    data = DataImport(args.data_path, 5)

    summary_data = []
    labels_true = []
    labels_pred = []
    logits_pred = []
    all_test = []
    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        if args.fold != -1 and fold != args.fold:
            continue
        print(f"Evaluating Fold {fold}")

        model_import = dynamic_importer(args.model_path)

        # ---- base kwargs: KEEP EXACTLY YOUR NON-OPTUNA DEFAULTS ----
        kwargs = dict()
        if args.model_path == "unet.py" or args.model_path == "unet_crf.py":
            if args.attention:
                # expected f1: 0.7402015172114621
                kwargs['bottleneck_type'] = 'attention'
                kwargs = kwargs | {
                    'epochs': 128,
                    'lr': 0.0005,
                    'dropout_rate': 0.,
                    'weight_decay': 1e-07,
                    'num_layers': 8,
                    'features': 64,
                    'transformer_window_size': 200,
                    'transformer_layers': 2,
                    'loss_gamma': 1.5
                }
                heads_per_channel = 16
                kwargs['transformer_nhead'] = max(kwargs['features'] // heads_per_channel, 1)
                kwargs['embed_dim'] = kwargs['features']
            else:
                # expected f1: 0.694895
                kwargs['bottleneck_type'] = 'block'
                kwargs = kwargs | {
                    'epochs': 64,
                    'lr': 0.0005,
                    'dropout_rate': 0.1,
                    'weight_decay': 1e-06,
                    'num_layers': 8,
                    'features': 32
                }

            if args.epochs:
                kwargs['epochs'] = args.epochs
        else:
            kwargs = {}

        # ---- NESTED OPTUNA: inner 5-fold CV on outer-train to overwrite kwargs ----
        augment_factor = 1
        if args.optuna:
            print(f"Running nested Optuna for outer fold {fold} (inner 5-fold on outer-train)...")
            study = optuna.create_study(direction='maximize')
            
            wandb_kwargs = {
                "project": "hmc-epg-mosquito",
                "group": f"{args.model_name}_optuna_study",
                "name": f"outer_fold_{fold}_study",
                "tags": ["optuna", "study"]
            }
            wandbc = WeightsAndBiasesCallback(metric_name="macro_f1", wandb_kwargs=wandb_kwargs)

            study.optimize(
                lambda t: optuna_objective(
                    data,
                    args,
                    t,
                    outer_train_index=train_index,
                    outer_fold=fold,
                    inner_folds=5,
                    **kwargs
                ),
                n_trials=50,
                show_progress_bar=True,
                callbacks=[wandbc],
            )

            print(f"[Fold {fold}] Best params:", study.best_params)
            augment_factor = study.best_params.get("augment_factor", 1)

            # Overwrite kwargs used for the REAL outer-fold training/eval
            kwargs = apply_best_params_to_kwargs(study.best_params, kwargs)

            # If attention case changes features (etc), recompute dependent args
            if args.model_path == "unet.py" and args.attention:
                heads_per_channel = 16
                kwargs['transformer_nhead'] = max(kwargs['features'] // heads_per_channel, 1)
                kwargs['embed_dim'] = kwargs['features']

            # Save per-fold hyperparam search plot (avoid overwrite)
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.savefig(f"{args.model_name}_hyper_outer{fold}.png")
            plt.close()

        #kwargs = {'epochs': 128, 'num_layers': 6, 'n_conv_steps_per_block': 3, 'features': 96, 'embed_dim': 96, 'lr': 0.0001528325773887917, 'dropout_rate': 0.2546315931008927, 'weight_decay': 4.1530388051130336e-08, 'transformer_window_size': 200, 'transformer_layers': 2, 'transformer_nhead': 96 // 16}

        # ---- Now do your ORIGINAL outer fold train/test split + probes ----
        train_data = [data.raw_dfs[i] for i in train_index]
        test_data  = [data.raw_dfs[i] for i in test_index]
        train_data, _ = data.get_probes(train_data)
        test_data, test_names = data.get_probes(test_data)

        # ---- augmentation: keep original behavior, but in optuna-mode respect tuned augment_factor ----
        if args.augment:
            if args.optuna:
                augmented_train_data = build_augmented_dataset(train_data, size=len(train_data) * augment_factor)
            else:
                augmented_train_data = build_augmented_dataset(train_data)
            print(f"{len(augmented_train_data)} Training Probes with Augment")

        # ---- Create model with (possibly overwritten) kwargs and run your original training ----
        # Initialize wandb for standalone evaluation runs
        run = None
        if not args.optuna:
            run = wandb.init(
                project="hmc-epg-mosquito",
                group=f"{args.model_name}_evaluation",
                name=f"fold_{fold}",
                config={"fold": fold, "model": args.model_name, "augment_factor": augment_factor, **kwargs},
                reinit=True,
                tags=["evaluation"]
            )

        model = model_import.Model(save_path=args.save_path, **kwargs)
        print("Training Model...")

        if args.augment:
            final_train_data = augmented_train_data
        else:
            final_train_data = train_data

        print(final_train_data[0].columns)
        model.train(final_train_data, test_data, fold)

        # ---- EVERYTHING BELOW HERE: keep your original evaluation/report code unchanged ----
        print("Evaluating Model...")

        if args.post_process is None:
            predicted_labels = model.predict(test_data)

        elif args.post_process.lower() == "viterbi" or args.post_process.lower() == "v":
            _, logits = model.predict(test_data, return_logits=True)
            logits = [l.transpose() for l in logits]
            post_process = PostProcessor(train_data, model.inv_label_map)
            predicted_labels = [post_process.postprocess_viterbi(logit) for logit in logits]

        elif args.post_process.lower() == "smooth" or args.post_process.lower() == "s":
            _, logits = model.predict(test_data, return_logits=True)
            post_process = PostProcessor(train_data, model.inv_label_map)
            predicted_labels = [post_process.postprocess_smooth(logit.transpose()) for logit in logits]
        elif args.post_process.lower() == "smooth" or args.post_process.lower() == "s":
            _, logits = model.predict(test_data, return_logits=True)
            logits = [l.transpose() for l in logits]
            post_process = PostProcessor(train_data, model.inv_label_map)
            predicted_labels = [post_process.postprocess_smooth(logit) for logit in logits]
        else:
            print("Choose a valid (case insensitive) post-processing arguement: either V/Viterbi or S/Smooth. Terminating program")
            assert False

        _, logits = model.predict(test_data, return_logits=True)
        print("Logits shape:", logits[0].shape)
        logits_pred.extend([l for l in logits])
        all_test.extend(test_data)

        print("Generating Report...")
        true, pred, stats = generate_report(test_data, predicted_labels, test_names, args.save_path, args.model_name, fold)
        summary_data.append(stats)
        labels_true.extend(true)
        labels_pred.extend(pred)

        if run is not None:
            # We already track the epoch loss in model_eval, we can track the final macro-f1 here
            f1 = f1_score(true, pred, average="macro")
            acc = accuracy_score(true, pred)
            run.log({"eval/macro_f1": f1, "eval/accuracy": acc})
            run.finish()

        
    out_summary_data = pd.concat(summary_data)
    out_summary_data.to_csv(f"{args.save_path}/{args.model_name}_SummaryStats_by_fold.csv")

    # Calculate statistics across every dataset
    labels = sorted(np.unique(labels_true))
    all_precision, all_recall, all_fscore, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = None, zero_division=0)
    temp_dict = {"precision" : all_precision, 
                 "recall" : all_recall, 
                 "fscore" : all_fscore}
    out_dataframe = pd.DataFrame(temp_dict, index=labels).stack()
    out_dataframe.index = out_dataframe.index.map('{0[1]}_{0[0]}'.format)
    out_dataframe = out_dataframe.to_frame().T
    out_dataframe["accuracy"] = accuracy_score(labels_true, labels_pred)
    all_precision_micro, all_recall_micro, all_fscore_micro, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = "micro", zero_division=0)
    all_precision_macro, all_recall_macro, all_fscore_macro, _ = precision_recall_fscore_support(labels_true, labels_pred, 
                                                            labels=labels, average = "macro", zero_division=0)
    out_dataframe["precision_micro"] = all_precision_micro
    out_dataframe["recall_micro"] = all_recall_micro
    out_dataframe["fscore_micro"] = all_fscore_micro
    out_dataframe["precision_macro"] = all_precision_macro
    out_dataframe["recall_macro"] = all_recall_macro
    out_dataframe["fscore_macro"] = all_fscore_macro
    out_dataframe.to_csv(f"{args.save_path}/{args.model_name}_SummaryStats.csv")

    overall = ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, \
                                            normalize = 'true')
    overall.plot().figure_.savefig(rf"{args.save_path}/{args.model_name}_OverallConfusionMatrix.png")

    all_data = pd.DataFrame({'labels_true': labels_true,
                             'labels_pred': labels_pred})
    all_data.to_csv(f"{args.save_path}/{args.model_name}_allpredictions.csv")
    #generate_roc(all_test, logits_pred, args.save_path, args.model_name, "Overall")

if __name__ == "__main__":
    main()
