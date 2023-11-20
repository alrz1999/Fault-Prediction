import numpy as np

from pipeline.models import PipelineStage, StageData


class EmbeddingModelTrainingStage(PipelineStage):
    def __init__(self, embedding_cls, dataset_name, embedding_dimension=50, stage_data=None, perform_export=False,
                 input_key=StageData.Keys.LINE_LEVEL_TOKENS):
        super().__init__(stage_data, perform_export)
        self.embedding_cls = embedding_cls
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension
        self.input_key = input_key

    def export_result(self):
        if self.result is None:
            raise Exception("Output data is not ready for exporting")

        self.result.export_model()

    def process(self):
        if self.input_key == StageData.Keys.LINE_LEVEL_TOKENS:
            data = self.stage_data[self.input_key]
        elif self.input_key == StageData.Keys.FILE_LEVEL_DF:
            data = self.stage_data[self.input_key]['SRC'].tolist()
        else:
            raise Exception()
        model = self.embedding_cls.train(
            data,
            dataset_name=self.dataset_name,
            embedding_dimension=self.embedding_dimension
        )

        self.result = model
        self.stage_data[StageData.Keys.EMBEDDING_MODEL] = self.result


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
        self.stage_data[StageData.Keys.EMBEDDING_MODEL] = self.result


class EmbeddingAdderStage(PipelineStage):
    def process(self):
        embedding_model = self.stage_data[StageData.Keys.EMBEDDING_MODEL]
        df = self.stage_data[StageData.Keys.FILE_LEVEL_DF]
        embeddings = embedding_model.text_to_vec(df['SRC'])
        print("embedding_shape:", np.array(embeddings).shape)
        self.result = embeddings
        self.stage_data[StageData.Keys.EMBEDDING] = self.result


class IndexToVecMatrixAdderStage(PipelineStage):
    def __init__(self, word_to_index_dict=None):
        super().__init__()
        self.word_to_index_dict = word_to_index_dict

    def process(self):
        embedding_model = self.stage_data[StageData.Keys.EMBEDDING_MODEL]
        word_index = embedding_model.get_word_to_index_dict() if self.word_to_index_dict is None else self.word_to_index_dict
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dimension()
        embedding_matrix = embedding_model.get_index_to_vec_matrix(word_index, vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.result = np.array(embedding_matrix)
            self.stage_data[StageData.Keys.INDEX_TO_VEC_MATRIX] = self.result
