import pandas as pd

import classification.keras_classifier.classifiers as keras_classifiers
from classification.BoW.BoW_baseline import BOWBaseLineClassifier
from classification.mlp.mlp_baseline import MLPBaseLineClassifier


def csv_to_latex_table(csv_file_path):
    """
    Reads a CSV file and generates LaTeX table code.

    Parameters:
    csv_file_path (str): The path to the CSV file.

    Returns:
    str: LaTeX table code.
    """

    column_name_mapping = {
        BOWBaseLineClassifier.__name__: 'BoW',
        MLPBaseLineClassifier.__name__: 'MLP',
        keras_classifiers.KerasCNNClassifier.__name__: 'CNN',
        keras_classifiers.KerasLSTMClassifier.__name__: 'LSTM',
        keras_classifiers.KerasBiLSTMClassifier.__name__: 'BiLSTM',
        keras_classifiers.KerasGRUClassifier.__name__: 'GRU',
        keras_classifiers.SiameseClassifier.__name__: 'Siamese',
        keras_classifiers.ReptileClassifier.__name__: 'Reptile',
        'Test Dataset': 'Test',
        'Train Dataset': 'Train',
    }
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Filter columns if specified
    if column_name_mapping:
        df = df[column_name_mapping.keys()]

    # Rename columns if mapping is provided
    if column_name_mapping:
        df.rename(columns=column_name_mapping, inplace=True)

    # Convert the DataFrame to LaTeX table code
    latex_table_code = df.to_latex(index=False, float_format="%.2f")

    return latex_table_code


def csv_to_latex_table_with_merged_cells(csv_file_path):
    """
    Reads a CSV file and generates LaTeX table code with merged cells for repeated values in 'Train Dataset'.

    Parameters:
    csv_file_path (str): The path to the CSV file.

    Returns:
    str: LaTeX table code.
    """
    column_name_mapping = {
        BOWBaseLineClassifier.__name__: 'BoW',
        MLPBaseLineClassifier.__name__: 'MLP',
        keras_classifiers.KerasCNNClassifier.__name__: 'CNN',
        keras_classifiers.KerasLSTMClassifier.__name__: 'LSTM',
        keras_classifiers.KerasBiLSTMClassifier.__name__: 'BiLSTM',
        keras_classifiers.KerasGRUClassifier.__name__: 'GRU',
        keras_classifiers.SiameseClassifier.__name__: 'Siamese',
        keras_classifiers.ReptileClassifier.__name__: 'Reptile',
        'Test Dataset': 'Test',
        'Train Dataset': 'Train',
    }
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Find unique values and their counts in 'Train Dataset'
    train_counts = df['Train Dataset'].value_counts()

    # Filter columns if specified
    if column_name_mapping:
        df = df[column_name_mapping.keys()]

    # Rename columns if mapping is provided
    if column_name_mapping:
        df.rename(columns=column_name_mapping, inplace=True)

    # Create a mapping for row span (number of times each value occurs)
    row_span = dict(zip(train_counts.index, train_counts))

    # Initialize an empty string to store LaTeX code
    begin = 'شروع'
    table = 'جدول'
    rule = 'خط‌پر'
    latex_code = f"\\{begin}{{{table}}}{{" + "c" * len(df.columns) + f"}}\n\\{rule}\n"

    # Add column headers
    latex_code += " & ".join(df.columns) + f" \\\\\n\\{rule}\n"

    # Track the current value of 'Train Dataset' for merging
    current_train_dataset = None
    for index, row in df.iterrows():
        if row['train'] != current_train_dataset:
            # New 'Train Dataset' value, start a new cell with row span
            current_train_dataset = row['train']
            latex_code += "\\multirow{" + str(row_span[current_train_dataset]) + "}{*}{" + current_train_dataset + "}"
        else:
            # Omit 'Train Dataset' value as it's part of a merged cell
            latex_code += "&"

        # Add other column values
        latex_code += " & ".join([str(row[col]) for col in df.columns if col != 'Train Dataset']) + " \\\\\n"

        # Add a line to separate rows
        if row['Train Dataset'] != df.iloc[min(index + 1, len(df) - 1)]['Train Dataset']:
            latex_code += "\\hline\n"

    # Add table end
    latex_code += "\\bottomrule\n\\end{tabular}"

    return latex_code
import pandas as pd

def csv_to_custom_latex_manual(csv_file_path):
    """
    Manually read a CSV file and generate LaTeX table format with custom tags and \lr{} for specified columns.

    Parameters:
    csv_file_path (str): Path to the CSV file.
    lr_columns (list): List of column names to apply \lr{}.
    custom_tags (dict): Custom tag replacements for LaTeX table elements.

    Returns:
    str: Manually generated LaTeX table code with custom tags.
    """
    column_name_mapping = {
        keras_classifiers.ReptileClassifier.__name__: 'Reptile',
        keras_classifiers.SiameseClassifier.__name__: 'Siamese',
        keras_classifiers.EnsembleClassifier.__name__: 'Ensemble',
        keras_classifiers.KerasCNNandGRUClassifier.__name__: 'CNN+GRU',
        keras_classifiers.KerasGRUClassifier.__name__: 'GRU',
        keras_classifiers.KerasBiLSTMClassifier.__name__: 'BiLSTM',
        keras_classifiers.KerasLSTMClassifier.__name__: 'LSTM',
        keras_classifiers.KerasCNNClassifier.__name__: 'CNN',
        # MLPBaseLineClassifier.__name__: 'MLP',
        # BOWBaseLineClassifier.__name__: 'BoW',
        'Test Dataset': 'Test',
        'Train Dataset': 'Train',
    }

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    df = df[df['Imbalance Method'] == 'smote']
    df.drop('Imbalance Method', inplace=True, axis=1)
    # Filter columns if specified
    if column_name_mapping:
        df = df[column_name_mapping.keys()]

    # Rename columns if mapping is provided
    if column_name_mapping:
        df.rename(columns=column_name_mapping, inplace=True)


    lr_columns = ['Train', 'Test']
    df = df.sort_values(by=lr_columns)

    begin = 'شروع'
    table = 'جدول'
    rule = 'خط‌پر'    # Begin the LaTeX table code
    end = 'پایان'
    font_size = 'scriptsize'
    tablet = 'لوح'
    alignment = 'تنظیم‌ازوسط'
    description = 'شرح'
    tag = 'برچسب'

    table_title = 'مقادیر MCC روش‌های مختلف در پیش‌بینی Cross-Release'
    latex_code = f"\\{begin}{{{tablet}}}[ht]\n"
    latex_code += f"\\{alignment}\n"
    latex_code += f"\\{description}{{{table_title}}}\n"

    latex_code += f"{{\\{font_size}\n"  # Begin font size environment
    latex_code += f"\\{begin}{{{table}}}{{|" + "c|" * len(df.columns) + f"}}\n\\{rule}\n"
    latex_code += " & ".join(df.columns) + f" \\\\\n\\{rule}\n"

    # Add data rows
    for _, row in df.iterrows():
        row_data = []
        for col in df.columns:
            cell = row[col]
            cell = f"\\lr{{{cell}}}"
            row_data.append(str(cell))
        latex_code += " & ".join(row_data) + " \\\\\n" + f"\n\\{rule}\n"

    # End the LaTeX table code
    latex_code += f"\\{end}{{{table}}}\n"
    latex_code += "}\n"  # End font size environment

    latex_code += f"\\{tag}{{{table}:{table_title}}}"
    latex_code += f"\\{end}{{{tablet}}}"
    return latex_code



if __name__ == '__main__':
    print(csv_to_custom_latex_manual('metrics/mcc.csv'))
