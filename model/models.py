from model.evaluation.evaluation import evaluate
import pandas as pd


class ClassifierModel:
    def do_evaluate(self, project):
        for rel in project.get_eval_releases():
            csv_file_path = self.get_result_dataset_path(rel.release_name)
            # csv_file_path = os.path.join(MLP_SAVE_PREDICTION_DIR, rel.release_name + '.csv')
            result_df = pd.read_csv(csv_file_path)
            true_labels = result_df['file-level-ground-truth']
            predicted_labels = result_df['prediction-label']
            evaluate(true_labels, predicted_labels)

    def train(self):
        raise NotImplementedError()

    def import_model(self):
        raise NotImplementedError()

    def export_model(self):
        raise NotImplementedError()

    def get_save_model_path(self):
        raise NotImplementedError()

    def get_result_dataset_path(self, dataset_name):
        raise NotImplementedError()
