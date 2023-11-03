import os
from config import WORD2VEC_DIR


class EmbeddingModel:
    def export_model(self):
        if not os.path.exists(WORD2VEC_DIR):
            os.makedirs(WORD2VEC_DIR)

    def import_model(self):
        raise NotImplementedError()

    def train_model(self):
        raise NotImplementedError()

    def get_save_path(self):
        raise NotImplementedError()
