import pandas as pd

from model.models import ClassifierModel
from pipeline.pipeline import PipelineStage, log_method_execution


class TrainingClassifierStage(PipelineStage):
    def __init__(self, classifier_cls, dataset_name, training_metadata=None, input_data=None, import_data=False,
                 export_data=False,
                 file_path=None):
        super().__init__(input_data, import_data, export_data, file_path)
        self.training_metadata = training_metadata
        self.classifier_cls = classifier_cls
        self.dataset_name = dataset_name

    @log_method_execution
    def import_output(self):
        if not self.import_data:
            return

        self.output_data = self.classifier_cls.import_model(self.dataset_name)
        return self.output_data

    @log_method_execution
    def export_output(self):
        if not self.export_data:
            return

        if self.output_data is None:
            raise Exception("Output data is not ready for exporting")

        self.output_data.export_model()

    @log_method_execution
    def process(self):
        model = self.classifier_cls.train(self.input_data, self.dataset_name, training_metadata=self.training_metadata)
        self.output_data = model
        return model


class PredictingClassifierStage(PipelineStage):
    def __init__(self, classifier: ClassifierModel, dataset_name, prediction_metadata=None, output_columns=None,
                 new_columns=None,
                 input_data=None, import_data=False, export_data=False,
                 file_path=None):
        super().__init__(input_data, import_data, export_data, file_path)
        self.classifier = classifier
        self.dataset_name = dataset_name
        self.prediction_metadata = prediction_metadata
        self.output_columns = output_columns
        self.new_columns = new_columns

    @log_method_execution
    def export_output(self):
        if not self.export_data:
            return

        if self.output_data is None:
            raise Exception("Output data is not ready for exporting")

        self.output_data.to_csv(self.classifier.get_result_dataset_path(self.dataset_name), index=False)

    @log_method_execution
    def import_output(self):
        if not self.import_data:
            return

        self.output_data = pd.read_csv(self.classifier.get_result_dataset_path(self.dataset_name))
        return self.output_data

    @log_method_execution
    def process(self):
        self.output_data = self.input_data.copy()
        predicted_labels = self.classifier.predict(self.input_data, prediction_metadata=self.prediction_metadata)
        if self.output_columns is not None:
            self.output_data = self.output_data[self.output_columns]
        if self.new_columns is not None and len(self.new_columns) != 0:
            for key, val in self.new_columns.items():
                self.output_data[key] = [val] * len(predicted_labels)
        self.output_data['prediction-label'] = predicted_labels
        return self.output_data
