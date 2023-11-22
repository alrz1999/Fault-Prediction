from classification.evaluation.evaluation import evaluate
from pipeline.models import PipelineStage, StageData


class EvaluationStage(PipelineStage):
    def __init__(self):
        super().__init__()

    def process(self):
        df = self.stage_data[StageData.Keys.PREDICTION_RESULT_DF.value]
        true_labels = df['label']
        predicted_labels = df['predicted_labels']
        evaluate(true_labels, predicted_labels)
