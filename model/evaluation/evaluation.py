import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


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


def main():
    # Replace these arrays with your actual test data and predictions
    true_labels = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    predicted_labels = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    evaluate(true_labels, predicted_labels)


if __name__ == '__main__':
    main()
