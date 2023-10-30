import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, roc_curve

def evaluate(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    confusion = confusion_matrix(true_labels, predicted_labels)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:\n", confusion)

    # Calculate the AUC-ROC (for binary classification)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    # Calculate Balanced Accuracy
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
    print("Balanced Accuracy:", balanced_acc)

    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    print("Matthews Correlation Coefficient (MCC):", mcc)


def main():
    # Replace these arrays with your actual test data and predictions
    true_labels = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    predicted_labels = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    evaluate(true_labels, predicted_labels)


if __name__ == '__main__':
    main()
