from classification.models import ClassifierModel
from pipeline.models import PipelineStage, StageData


class ClassifierTrainingStage(PipelineStage):
    def __init__(self, classifier_cls, dataset_name, perform_export=False):
        super().__init__(perform_export=perform_export)
        self.classifier_cls = classifier_cls
        self.dataset_name = dataset_name

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.export_model()

    def process(self):
        train_data = self.stage_data[StageData.Keys.TRAINING_SOURCE_CODE_DF.value]
        validation_data = self.stage_data.get(StageData.Keys.VALIDATION_SOURCE_CODE_DF.value)

        model = self.classifier_cls.train(
            source_code_df=train_data,
            dataset_name=self.dataset_name,
            metadata=self.stage_data,
            validation_source_code_df=validation_data
        )
        self.result = model
        self.stage_data[StageData.Keys.CLASSIFIER_MODEL.value] = self.result


class PredictingClassifierStage(PipelineStage):
    def __init__(self, dataset_name, output_columns=None, new_columns=None, perform_export=False):
        super().__init__(perform_export=perform_export)
        self.dataset_name = dataset_name
        self.output_columns = output_columns
        self.new_columns = new_columns

    def get_classifier(self) -> ClassifierModel:
        return self.stage_data[StageData.Keys.CLASSIFIER_MODEL.value]

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.to_csv(self.get_classifier().get_result_dataset_path(self.dataset_name), index=False)

    def process(self):
        data = self.stage_data[StageData.Keys.EVALUATION_SOURCE_CODE_DF.value]
        predicted_probabilities = self.get_classifier().predict(data, metadata=self.stage_data)
        if self.output_columns is not None:
            data = data[self.output_columns]
        if self.new_columns is not None and len(self.new_columns) != 0:
            for key, val in self.new_columns.items():
                data[key] = [val] * len(predicted_probabilities)
        data['predicted_probabilities'] = predicted_probabilities
        self.result = data
        self.stage_data[StageData.Keys.PREDICTION_RESULT_DF.value] = self.result
