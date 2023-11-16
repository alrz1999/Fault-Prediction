from model.evaluation.evaluation import evaluate
from pipeline.pipeline import PipelineStage, log_method_execution


class EvaluationStage(PipelineStage):
    def __init__(self):
        super().__init__()

    @log_method_execution
    def process(self):
        df = self.input_data
        true_labels = df['Bug']
        predicted_labels = df['prediction-label']
        evaluate(true_labels, predicted_labels)
        self.output_data = self.input_data
        return self.output_data
