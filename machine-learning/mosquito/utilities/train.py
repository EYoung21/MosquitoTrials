import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from data_augmentation import build_augmented_dataset
from utilities.model_factory import build_model
from sklearn.metrics import precision_recall_fscore_support, \
                            confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            accuracy_score, \
                            f1_score
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

def run_experiment(cfg: DictConfig, data):
    labels_true = []
    labels_pred = []
    


    for fold, (train_index, test_index) in enumerate(data.cross_val_iter):
        if cfg.train.folds != -1 and fold != cfg.train.folds:
            continue
        print(f"Evaluating Fold {fold}")

        model, model_conf = build_model(cfg.model)
        train_data = [data.raw_dfs[i] for i in train_index]
        test_data  = [data.raw_dfs[i] for i in test_index]
        train_data, _ = data.get_probes(train_data)
        test_data, test_names = data.get_probes(test_data)
        if cfg.train.augment:
            train_data = build_augmented_dataset(train_data, size=len(train_data) * cfg.train.augment_factor)
            print(f"{len(train_data)} Training Probes with Augment")
        
        model.train(train_data, test_data, train_cfg = cfg.train)

        predicted_labels = model.predict(test_data)

        f1 = f1_score(labels_true, labels_pred, average="macro")
        return f1




