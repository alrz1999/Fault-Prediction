import os

import gensim
from gensim.models import Word2Vec

from config import WORD2VEC_DIR, DATE_SAVE_DIR


def export_word2vec_model(project, embedding_dim=50):
    w2v_path = WORD2VEC_DIR

    save_path = os.path.join(w2v_path, project.name + '-' + str(embedding_dim) + 'dim.bin')

    if os.path.exists(save_path):
        print('word2vec model at {} is already exists'.format(save_path))
        return

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    word2vec = train_word2vec(project, embedding_dim)

    word2vec.save(save_path)
    print('save word2vec model at path {} done'.format(save_path))


def import_word2vec_model(project, embedding_dim=50):
    w2v_path = WORD2VEC_DIR
    save_path = os.path.join(w2v_path, project.name + '-' + str(embedding_dim) + 'dim.bin')
    return gensim.models.Word2Vec.load(save_path)


def train_word2vec(project, embedding_dim):
    all_texts = project.get_train_release().get_all_lines(DATE_SAVE_DIR)
    word2vec = Word2Vec(all_texts, vector_size=embedding_dim, min_count=1, sorted_vocab=1)
    return word2vec

# p = sys.argv[1]
# export_word2vec_model(Project(p), 50)
# p = 'activemq'
# t = get_all_texts(p)
# print(type(t), len(t))
