import os

import gensim
import numpy as np
from gensim.models import Word2Vec

from config import WORD2VEC_DIR
from embedding.models import EmbeddingModel


class GensimWord2VecModel(EmbeddingModel):
    def __init__(self, model, dataset_name, embedding_dimension):
        self.model = model
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension

    def export_model(self):
        save_path = self.get_model_save_path(self.dataset_name, embedding_dimension=self.embedding_dimension)
        if os.path.exists(save_path):
            print('word2vec model at {} is already exists'.format(save_path))

        self.model.save(save_path)
        print('save word2vec model at path {} done'.format(save_path))

    @classmethod
    def import_model(cls, dataset_name, **kwargs):
        save_path = cls.get_model_save_path(dataset_name, **kwargs)
        embedding_dimension = kwargs.get('embedding_dimension', 50)
        model = gensim.models.Word2Vec.load(save_path)
        return GensimWord2VecModel(model, dataset_name, embedding_dimension)

    @classmethod
    def train(cls, data, **kwargs):
        embedding_dimension = kwargs.get('embedding_dimension', 50)
        dataset_name = kwargs.get('dataset_name')
        line_tokens = data
        model = Word2Vec(line_tokens, vector_size=embedding_dimension, min_count=1, sorted_vocab=1)
        return GensimWord2VecModel(model, dataset_name, embedding_dimension)

    @classmethod
    def get_model_save_path(cls, dataset_name, **kwargs):
        embedding_dimension = kwargs.get('embedding_dimension', 50)
        return os.path.join(WORD2VEC_DIR, dataset_name + '-' + str(embedding_dimension) + 'dim.bin')

    def get_embeddings(self, data):
        embeddings = []
        for code in data:
            code_embeddings = [
                self.model.wv[word] if word in self.model.wv else np.zeros(self.model.vector_size) for word
                in
                code.split()]
            code_embeddings = np.mean(code_embeddings, axis=0)
            embeddings.append(code_embeddings)
        return embeddings
