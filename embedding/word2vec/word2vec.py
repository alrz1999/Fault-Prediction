import os

import gensim
import numpy as np
from gensim.models import Word2Vec
from keras.src.preprocessing.text import Tokenizer

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
        return cls(model, dataset_name, embedding_dimension)

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

    def text_to_vec(self, texts):
        vecs = []
        for text in texts:
            vec = [
                self.model.wv[word] if word in self.model.wv else np.zeros(self.model.vector_size) for word
                in
                text.split()]
            vec = np.mean(vec, axis=0)
            vecs.append(vec)
        return vecs

    def text_to_indexes(self, texts):
        texts_indexes = []
        for text in texts:
            text_indexes = [
                self.model.wv.key_to_index[word] for word
                in
                text.split() if word in self.model.wv.key_to_index]
            texts_indexes.append(text_indexes)
        return texts_indexes

    def get_word_to_index_dict(self):
        return self.model.wv.key_to_index

    def get_index_to_vec_matrix(self, word_index, vocab_size, embedding_dim):
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        for word in self.model.wv.index_to_key:
            if word in word_index:
                vec = self.model.wv[word]
                embedding_matrix[word_index[word]] = np.array(vec, dtype=np.float32)[:embedding_dim]

        return embedding_matrix

    def get_vocab_size(self):
        return len(self.model.wv.index_to_key)

    def get_embedding_dimension(self):
        return self.embedding_dimension


class KerasTokenizer(EmbeddingModel):
    def __init__(self, tokenizer, embedding_dimension):
        self.embedding_dimension = embedding_dimension
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, data, **kwargs):
        embedding_dimension = kwargs.get('embedding_dimension', 50)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)
        return cls(tokenizer, embedding_dimension)

    def text_to_indexes(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def get_word_to_index_dict(self):
        return self.tokenizer.word_index

    def get_index_to_vec_matrix(self, word_index, vocab_size, embedding_dim):
        return None

    def get_vocab_size(self):
        return len(self.get_word_to_index_dict()) + 1

    def get_embedding_dimension(self):
        return self.embedding_dimension
