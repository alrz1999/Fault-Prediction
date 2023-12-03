import numpy as np

from pipeline.models import PipelineStage, StageData


class EmbeddingModelTrainingStage(PipelineStage):
    def __init__(self, embedding_cls, dataset_name, embedding_dim, token_extractor,
                 stage_data=None, perform_export=False):
        super().__init__(stage_data, perform_export)
        self.embedding_cls = embedding_cls

        self.dataset_name = dataset_name
        self.stage_data.add_data('dataset_name', dataset_name)

        self.embedding_dim = embedding_dim
        self.stage_data.add_data('embedding_dim', embedding_dim)

        self.token_extractor = token_extractor
        self.stage_data.add_data('token_extractor', token_extractor)

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.export_model()

    def process(self):
        texts = self.stage_data[StageData.Keys.TRAINING_SOURCE_CODE_DF.value]['text'].tolist()

        model = self.embedding_cls.train(
            texts,
            self.stage_data
        )

        self.result = model
        self.stage_data[StageData.Keys.EMBEDDING_MODEL.value] = self.result


class EmbeddingModelImporterStage(PipelineStage):
    def __init__(self, embedding_cls, dataset_name, embedding_dim, token_extractor):
        super().__init__()
        self.embedding_cls = embedding_cls

        self.dataset_name = dataset_name
        self.stage_data.add_data('dataset_name', dataset_name)

        self.embedding_dim = embedding_dim
        self.stage_data.add_data('embedding_dim', embedding_dim)

        self.token_extractor = token_extractor
        self.stage_data.add_data('token_extractor', token_extractor)

    def import_model(self):
        model = self.embedding_cls.import_model(
            self.dataset_name,
            self.stage_data
        )

        return model

    def process(self):
        self.result = self.import_model()
        self.stage_data[StageData.Keys.EMBEDDING_MODEL.value] = self.result


class IndexToVecMatrixAdderStage(PipelineStage):
    def __init__(self, word_to_index_dict=None):
        super().__init__()
        self.word_to_index_dict = word_to_index_dict

    def process(self):
        if StageData.Keys.EMBEDDING_MODEL.value not in self.stage_data:
            return
        embedding_model = self.stage_data[StageData.Keys.EMBEDDING_MODEL.value]
        word_index = embedding_model.get_word_to_index_dict() if self.word_to_index_dict is None else self.word_to_index_dict
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dim()
        embedding_matrix = embedding_model.get_embedding_matrix(word_index, vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.result = np.array(embedding_matrix)
            self.stage_data[StageData.Keys.EMBEDDING_MATRIX.value] = self.result
