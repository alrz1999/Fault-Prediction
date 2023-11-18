import os
from config import WORD2VEC_DIR


class EmbeddingModel:
    def export_model(self):
        if not os.path.exists(WORD2VEC_DIR):
            os.makedirs(WORD2VEC_DIR)

    @classmethod
    def import_model(cls, dataset_name, **kwargs):
        raise NotImplementedError()

    @classmethod
    def train(cls, data, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_model_save_path(cls, dataset_name, **kwargs):
        raise NotImplementedError()

    def get_embeddings(self, data):
        raise NotImplementedError()

    def get_embedding_matrix(self, word_index, embedding_dim):
        raise NotImplementedError()
