import os

import pandas as pd

from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR, MLP_SAVE_PREDICTION_DIR
from data.models import Project
from model.evaluation.evaluation import evaluate
from model.mlp.mlp_baseline import train_model, predict_defective_files_in_releases


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    # train_and_generate_output(project)
    do_evaluate(project)


def do_evaluate(project):
    for rel in project.get_eval_releases():
        csv_file_path = os.path.join(MLP_SAVE_PREDICTION_DIR, rel.release_name + '.csv')
        result_df = pd.read_csv(csv_file_path)
        true_labels = result_df['file-level-ground-truth']
        predicted_labels = result_df['prediction-label']
        evaluate(true_labels, predicted_labels)


def train_and_generate_output(project):
    train_release = project.get_train_release()
    eval_releases = project.get_eval_releases()
    train_model(train_release)
    predict_defective_files_in_releases(train_release, eval_releases)


if __name__ == '__main__':
    main()
