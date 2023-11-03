import os

import gensim
from gensim.models import Word2Vec

from config import WORD2VEC_DIR
from data.models import LineLevelDatasetGenerator
from embedding.models import EmbeddingModel


class GensimWord2VecModel(EmbeddingModel):
    def __init__(self, line_level_dataset_generator, dataset_name, embedding_dimension=50):
        self.line_level_dataset_generator: LineLevelDatasetGenerator = line_level_dataset_generator
        self.dataset_name = dataset_name
        self.embedding_dimension = embedding_dimension

    def export_model(self):
        save_path = self.get_save_path()
        if os.path.exists(save_path):
            print('word2vec model at {} is already exists'.format(save_path))

        model = self.train_model()
        model.save(save_path)
        print('save word2vec model at path {} done'.format(save_path))

    def import_model(self):
        return gensim.models.Word2Vec.load(self.get_save_path())

    def train_model(self):
        all_texts = self.line_level_dataset_generator.get_all_lines_tokens()
        word2vec = Word2Vec(all_texts, vector_size=self.embedding_dimension, min_count=1, sorted_vocab=1)
        return word2vec

    def get_save_path(self):
        return os.path.join(WORD2VEC_DIR, self.dataset_name + '-' + str(self.embedding_dimension) + 'dim.bin')

# p = sys.argv[1]
# export_word2vec_model(Project(p), 50)
# p = 'activemq'
# t = get_all_texts(p)
# print(type(t), len(t))
