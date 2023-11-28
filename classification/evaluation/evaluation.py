import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    balanced_accuracy_score, matthews_corrcoef, roc_curve, auc


def evaluate(true_labels, predicted_probabilities):
    predicted_labels = list([round(x) for x in predicted_probabilities])
    print(f'predicted_labels = {predicted_labels}')

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
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


def main():
    # Replace these arrays with your actual test data and predictions
    true_labels = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    predicted_labels = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    evaluate(true_labels, predicted_labels)


if __name__ == '__main__':
    main()
