import os

import gensim
import numpy as np
from gensim.models import Word2Vec
from keras.src.layers import TextVectorization
from keras.src.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer

from config import WORD2VEC_DIR
from embedding.preprocessing.token_extraction import TokenExtractor


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

    def get_vocab_size(self):
        raise NotImplementedError()

    def get_embedding_dim(self):
        raise NotImplementedError()

    def get_embedding_matrix(self, word_index, vocab_size, embedding_dim):
        raise NotImplementedError()


class GensimWord2VecModel(EmbeddingModel):
    def __init__(self, model, dataset_name, embedding_dim, token_extractor):
        self.model = model
        self.dataset_name = dataset_name
        self.embedding_dim = embedding_dim
        self.token_extractor: TokenExtractor = token_extractor

    def export_model(self):
        save_path = self.get_model_save_path(self.dataset_name, {'embedding_dim': self.embedding_dim})
        if os.path.exists(save_path):
            print('word2vec model at {} is already exists'.format(save_path))

        self.model.save(save_path)
        print('save word2vec model at path {} done'.format(save_path))

    @classmethod
    def import_model(cls, dataset_name, metadata):
        save_path = cls.get_model_save_path(dataset_name, metadata)
        embedding_dim = metadata.get('embedding_dim', 50)
        token_extractor = metadata.get('token_extractor')
        model = gensim.models.Word2Vec.load(save_path)
        return cls(model, dataset_name, embedding_dim, token_extractor)

    @classmethod
    def train(cls, texts, metadata):
        embedding_dim = metadata.get('embedding_dim')
        dataset_name = metadata.get('dataset_name')
        token_extractor: TokenExtractor = metadata.get('token_extractor')

        tokens = [token_extractor.extract_tokens(text) for text in texts]
        tokens.extend([['<pad>'] * 100000])
        model = Word2Vec(tokens, vector_size=embedding_dim, min_count=1, sorted_vocab=1)
        output_model = cls(model, dataset_name, embedding_dim, token_extractor)
        print(f"{cls.__name__} training on {len(texts)} texts finished with {output_model.get_vocab_size()} vocab_size")
        return output_model

    @classmethod
    def get_model_save_path(cls, dataset_name, metadata):
        embedding_dim = metadata.get('embedding_dim')
        return os.path.join(WORD2VEC_DIR, dataset_name + '-' + str(embedding_dim) + 'dim.bin')

    def text_to_vec(self, texts):
        vecs = []
        for text in texts:
            tokens = self.token_extractor.extract_tokens(text)
            if len(tokens) > 0:
                vec = [self.model.wv[word] if word in self.model.wv else np.zeros(self.model.vector_size) for word in
                       tokens]
            else:
                vec = [np.zeros(self.model.vector_size)]
            vec = np.mean(vec, axis=0)
            vecs.append(vec)
        return vecs

    def text_to_indexes(self, texts):
        texts_indexes = []
        for text in texts:
            text_indexes = [
                self.model.wv.key_to_index[word] for word
                in
                self.token_extractor.extract_tokens(text) if word in self.model.wv.key_to_index]
            texts_indexes.append(text_indexes)
        return texts_indexes

    def get_word_to_index_dict(self):
        return self.model.wv.key_to_index

    def get_vocab_size(self):
        return len(self.model.wv.index_to_key)

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_embedding_matrix(self, word_index, vocab_size, embedding_dim):
        embedding_matrix = np.zeros((vocab_size, embedding_dim))

        present_words = 0
        for word in self.model.wv.index_to_key:
            if word in word_index:
                present_words += 1
                vec = self.model.wv[word]
                embedding_matrix[word_index[word]] = np.array(vec, dtype=np.float32)[:embedding_dim]
        absent_words = len(word_index) - present_words
        print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)),
              '% of total words')
        return embedding_matrix


class KerasTokenizer(EmbeddingModel):
    def __init__(self, tokenizer, embedding_dim):
        self.embedding_dim = embedding_dim
        self.tokenizer = tokenizer

    @classmethod
    def train(cls, texts, metadata):
        embedding_dim = metadata.get('embedding_dim')
        token_extractor: TokenExtractor = metadata.get('token_extractor')
        to_lowercase = metadata.get('to_lowercase')
        num_words = metadata.get('num_words')
        oov_token = metadata.get('oov_token')

        tokens = [token_extractor.extract_tokens(text) for text in texts]
        tokenizer = Tokenizer(lower=to_lowercase, num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(tokens)
        output_model = cls(tokenizer, embedding_dim)
        print(f"{cls.__name__} training on {len(texts)} texts finished with {output_model.get_vocab_size()} vocab_size")
        return output_model

    def text_to_indexes(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def get_word_to_index_dict(self):
        return self.tokenizer.word_index

    def get_vocab_size(self):
        return len(self.get_word_to_index_dict()) + 1

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_embedding_matrix(self, word_index, vocab_size, embedding_dim):
        return None


class SklearnCountTokenizer(EmbeddingModel):
    def __init__(self, count_vectorizer, embedding_dim):
        self.count_vectorizer: CountVectorizer = count_vectorizer
        self.embedding_dim = embedding_dim

    @classmethod
    def train(cls, texts, metadata):
        embedding_dim = metadata.get('embedding_dim')
        to_lowercase = metadata.get('to_lowercase')
        vectorizer = CountVectorizer(lowercase=to_lowercase)
        vectorizer.fit(texts)
        output_model = cls(vectorizer, embedding_dim)
        print(f"{cls.__name__} training on {len(texts)} texts finished with {output_model.get_vocab_size()} vocab_size")
        return output_model

    def text_to_indexes(self, texts):
        return self.count_vectorizer.transform(texts).toarray()

    def get_vocab_size(self):
        return len(self.count_vectorizer.vocabulary_)

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_word_to_index_dict(self):
        return None

    def get_embedding_matrix(self, word_index, vocab_size, embedding_dim):
        return None


class KerasTextVectorizer(EmbeddingModel):
    def __init__(self, vectorize_layer, embedding_dim):
        self.vectorize_layer: TextVectorization = vectorize_layer
        self.embedding_dim = embedding_dim

    @classmethod
    def train(cls, texts, metadata):
        embedding_dim = metadata.get('embedding_dim')
        max_tokens = metadata.get('vocab_size')
        output_sequence_length = metadata.get('max_seq_len')

        vectorize_layer = TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=output_sequence_length,
        )
        vectorize_layer.adapt(texts)
        output_model = cls(vectorize_layer, embedding_dim)
        print(f"{cls.__name__} training on {len(texts)} texts finished with {output_model.get_vocab_size()} vocab_size")
        return output_model

    def text_to_indexes(self, texts):
        # text = tf.expand_dims(text, -1)
        return self.vectorize_layer(texts)

    def get_vocab_size(self):
        return self.vectorize_layer.vocabulary_size()

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_word_to_index_dict(self):
        return None

    def get_embedding_matrix(self, word_index, vocab_size, embedding_dim):
        return None
