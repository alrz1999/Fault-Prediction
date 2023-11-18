from classification.models import ClassifierModel
from pipeline.models import PipelineStage


class ClassifierTrainingStage(PipelineStage):
    def __init__(self, classifier_cls, dataset_name, training_metadata=None, perform_export=False):
        super().__init__(perform_export=perform_export)
        self.training_metadata = training_metadata
        self.classifier_cls = classifier_cls
        self.dataset_name = dataset_name

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.export_model()

    def process(self):
        train_data = self.stage_data['train_data']
        model = self.classifier_cls.train(train_data, self.dataset_name, training_metadata=self.training_metadata)
        self.result = model
        self.stage_data['classifier_model'] = self.result


class PredictingClassifierStage(PipelineStage):
    def __init__(self, classifier: ClassifierModel, dataset_name, prediction_metadata=None, output_columns=None,
                 new_columns=None, perform_export=False):
        super().__init__(perform_export=perform_export)
        self.classifier = classifier
        self.dataset_name = dataset_name
        self.prediction_metadata = prediction_metadata
        self.output_columns = output_columns
        self.new_columns = new_columns

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.to_csv(self.classifier.get_result_dataset_path(self.dataset_name), index=False)

    def process(self):
        data = self.stage_data['test_data'].copy()
        predicted_labels = self.classifier.predict(data, prediction_metadata=self.prediction_metadata)
        if self.output_columns is not None:
            data = data[self.output_columns]
        if self.new_columns is not None and len(self.new_columns) != 0:
            for key, val in self.new_columns.items():
                data[key] = [val] * len(predicted_labels)
        data['prediction-label'] = predicted_labels
        self.result = data
        self.stage_data['prediction_result'] = self.result
