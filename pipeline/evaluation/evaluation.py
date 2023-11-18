from classification.evaluation.evaluation import evaluate
from pipeline.models import PipelineStage


class EvaluationStage(PipelineStage):
    def __init__(self):
        super().__init__()

    def process(self):
        df = self.stage_data['ready_for_evaluation_df']
        true_labels = df['Bug']
        predicted_labels = df['prediction-label']
        evaluate(true_labels, predicted_labels)
