import os
from config import WORD2VEC_DIR


class EmbeddingModel:
    def export_model(self):
        if not os.path.exists(WORD2VEC_DIR):
            os.makedirs(WORD2VEC_DIR)

    @classmethod
    def import_model(cls, dataset_name, metadata):
        raise NotImplementedError()

    @classmethod
    def train(cls, texts, metadata):
        raise NotImplementedError()

    @classmethod
    def get_model_save_path(cls, dataset_name, metadata):
        raise NotImplementedError()

    def text_to_vec(self, data):
        raise NotImplementedError()

    def text_to_indexes(self, texts):
        raise NotImplementedError()

    def get_word_to_index_dict(self):
        raise NotImplementedError()

    def get_index_to_vec_matrix(self, word_index, vocab_size, embedding_dim):
        raise NotImplementedError()

    def get_vocab_size(self):
        raise NotImplementedError()

    def get_embedding_dim(self):
        raise NotImplementedError()
