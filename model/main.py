from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR
from data.models import Project
from model.custom.custom_model import KerasClassifier
from model.mlp.mlp_baseline import MLPBaseLineClassifier
from model.BoW.BoW_baseline import BOWBaseLineClassifier


def mlp_classifier(project):
    train_release = project.get_train_release()
    eval_releases = project.get_eval_releases()
    baseline_classifier = MLPBaseLineClassifier(train_release)
    # baseline_classifier.train()
    # baseline_classifier.predict_defective_files_in_releases(eval_releases)
    baseline_classifier.do_evaluate(project)


def bow_classifier(project):
    train_release = project.get_train_release()
    eval_releases = project.get_eval_releases()
    baseline_classifier = BOWBaseLineClassifier(train_release)
    # baseline_classifier.train()
    # baseline_classifier.predict_defective_files_in_releases(eval_releases)
    baseline_classifier.do_evaluate(project)


def keras_classifier(project):
    c = KerasClassifier(project.get_train_release())
    c.train()
    c.evaluate(project.get_output_dataset(), batch_size=32)


if __name__ == '__main__':
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    # keras_classifier(project)
    # bow_classifier(project)
    # mlp_classifier(project)
