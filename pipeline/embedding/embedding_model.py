from pipeline.models import PipelineStage


class EmbeddingModelTrainingStage(PipelineStage):
    def __init__(self, embedding_cls, dataset_name, embedding_dimension=50, stage_data=None, perform_export=False):
        super().__init__(stage_data, perform_export)
        self.embedding_cls = embedding_cls
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.export_model()

    def process(self):
        model = self.embedding_cls.train(
            self.stage_data['embedding_input'],
            dataset_name=self.dataset_name,
            embedding_dimension=self.embedding_dimension
        )

        self.result = model
        self.stage_data['embedding_model'] = self.result


class EmbeddingModelImporterStage(PipelineStage):
    def __init__(self, embedding_cls, dataset_name, embedding_dimension=50):
        super().__init__()
        self.embedding_cls = embedding_cls
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension

    def import_model(self):
        model = self.embedding_cls.import_model(
            self.dataset_name,
            embedding_dimenstion=self.embedding_dimension
        )

        return model

    def process(self):
        self.result = self.import_model()
        self.stage_data['embedding_model'] = self.result
