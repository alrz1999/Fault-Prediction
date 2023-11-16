from embedding.models import EmbeddingModel
from pipeline.pipeline import PipelineStage, log_method_execution
import pandas as pd


class TrainingEmbeddingModelStage(PipelineStage):
    def __init__(self, embedding_cls, dataset_name, embedding_dimension=50, input_data=None, import_data=False,
                 export_data=False, file_path=None):
        super().__init__(input_data, import_data, export_data, file_path)
        self.embedding_cls = embedding_cls
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension

    @log_method_execution
    def import_output(self):
        if not self.import_data:
            return

        self.output_data = self.embedding_cls.import_model(
            self.dataset_name,
            embedding_dimenstion=self.embedding_dimension
        )

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
        model = self.embedding_cls.train(
            self.input_data,
            dataset_name=self.dataset_name,
            embedding_dimension=self.embedding_dimension
        )
        self.output_data = model
        return model


class EmbeddingColumnAdderStage(PipelineStage):
    def __init__(self, embedding_model: EmbeddingModel, input_data=None):
        super().__init__(input_data, False, False, None)
        self.embedding_model = embedding_model

    @log_method_execution
    def import_output(self):
        raise Exception("Import is not valid")

    @log_method_execution
    def export_output(self):
        raise Exception("Export is not valid")

    @log_method_execution
    def process(self):
        df = self.input_data
        embeddings = self.embedding_model.get_embeddings(df['SRC'])
        self.output_data = df.copy()
        self.output_data['embedding'] = pd.Series(embeddings)
        return self.output_data
