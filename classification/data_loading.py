import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(dataset_path, encoding='latin'):
    """
    Load a dataset from a CSV file and split it into features and labels.

    Args:
        dataset_path (str): Path to the CSV dataset file.
        encoding (str)
    Returns:
        features (pd.DataFrame): DataFrame containing the features.
        labels (pd.Series): Series containing the labels.
    """
    df = pd.read_csv(dataset_path, encoding=encoding)  # Load the dataset from the CSV file

    # Assume the last column is the target variable (label) and the rest are features
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]

    return features, labels


def split_data(features, labels, test_size=0.2, random_state=None):
    """
    Split the dataset into training and testing sets.

    Args:
        features (pd.DataFrame): DataFrame containing the features.
        labels (pd.Series): Series containing the labels.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        X_train (pd.DataFrame): Features for training.
        X_test (pd.DataFrame): Features for testing.
        y_train (pd.Series): Labels for training.
        y_test (pd.Series): Labels for testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test
