from model_eval import *
import argparse


def main():
    print("Running summarize_folds.py")
    parser = argparse.ArgumentParser(description='Summarize folds for model evaluation.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the summary statistics.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for evaluation.')
    args = parser.parse_args()

    PATH = args.save_path
    MODEL_NAME = args.model_name
    num_folds = args.num_folds

    labels_true, labels_pred = [], []
    for fold in range(num_folds):
        model_path = f"{PATH}/fold_{fold}/{MODEL_NAME}_allpredictions.csv"
        if not os.path.exists(model_path):
            continue
        df = pd.read_csv(model_path)
        labels_true.extend(df["labels_true"].tolist())
        labels_pred.extend(df["labels_pred"].tolist())

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
    os.makedirs(f"{PATH}/summary_stats", exist_ok=True)
    out_dataframe.to_csv(f"{PATH}/summary_stats/{MODEL_NAME}_SummaryStats.csv")

    overall = ConfusionMatrixDisplay.from_predictions(labels_true, labels_pred, \
                                            normalize = 'true')
    overall.plot().figure_.savefig(rf"{PATH}/summary_stats/{MODEL_NAME}_OverallConfusionMatrix.png")
    all_data = pd.DataFrame({'labels_true': labels_true,
                                'labels_pred': labels_pred})
    all_data.to_csv(f"{PATH}/summary_stats/{MODEL_NAME}_allpredictions.csv")

if __name__ == "__main__":
    main()