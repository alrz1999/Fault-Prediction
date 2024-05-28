from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, precision_recall_curve

import pandas as pd
import os

from config import RESULT_METRICS_DIR


def evaluate(true_labels, predicted_probabilities, train_dataset_name=None, test_dataset_name=None,
             classifier_name=None):
    predicted_probabilities = np.squeeze(predicted_probabilities)
    predicted_labels = np.round(predicted_probabilities)
    print(f'predicted_labels = {predicted_labels}')

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probabilities)

    # Plot ROC curve
    plt.figure(figsize=(12, 5))

    # Subplot for ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Subplot for Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    confusion = confusion_matrix(true_labels, predicted_labels)

    print("Confusion Matrix:\n", confusion)
    print(f"count: {len(true_labels)}")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print(f"AUC = {roc_auc:.2f}")
    # Calculate Balanced Accuracy
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
    print("Balanced Accuracy:", balanced_acc)

    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    print("Matthews Correlation Coefficient (MCC):", mcc)

    if train_dataset_name and test_dataset_name and classifier_name:
        store_metric('accuracy', train_dataset_name, test_dataset_name, classifier_name, accuracy)
        store_metric('precision', train_dataset_name, test_dataset_name, classifier_name, precision)
        store_metric('recall', train_dataset_name, test_dataset_name, classifier_name, recall)
        store_metric('f1', train_dataset_name, test_dataset_name, classifier_name, f1)
        store_metric('AUC', train_dataset_name, test_dataset_name, classifier_name, roc_auc)
        store_metric('balanced_acc', train_dataset_name, test_dataset_name, classifier_name, balanced_acc)
        store_metric('mcc', train_dataset_name, test_dataset_name, classifier_name, mcc)


def store_metric(metric_name, train_dataset_name, test_dataset_name, classifier_name, metric_value):
    Path(RESULT_METRICS_DIR).mkdir(parents=True, exist_ok=True)

    filename = os.path.join(RESULT_METRICS_DIR, f"{metric_name}.csv")

    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=[0, 1])
    else:
        df = pd.DataFrame(columns=['Train Dataset', 'Test Dataset', classifier_name])
        df.set_index(['Train Dataset', 'Test Dataset'], inplace=True)

    if classifier_name not in df.columns:
        df[classifier_name] = pd.Series(dtype='float64')

    # Check if the index exists, if not, create it
    if (train_dataset_name, test_dataset_name) not in df.index:
        df.loc[(train_dataset_name, test_dataset_name), :] = None

    df.at[(train_dataset_name, test_dataset_name), classifier_name] = round(metric_value, ndigits=2)
    df.to_csv(filename)