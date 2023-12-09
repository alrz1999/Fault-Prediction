import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Model
import nltk
import re
import matplotlib.pyplot as plt
import sys
import seaborn as sns

from keras.src.engine.base_layer import Layer
from keras.src.layers import Conv2D
from sklearn.metrics import roc_auc_score
from nltk import tokenize

from classification.utils import LineLevelToFileLevelDatasetMapper
from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project
from embedding.preprocessing.token_extraction import CustomTokenExtractor
from embedding.models import GensimWord2VecModel, KerasTokenizer


# https://www.kaggle.com/code/hsankesara/news-classification-using-han

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )

    line_level_dataset = project.get_train_release().get_processed_line_level_dataset()
    docs, labels = LineLevelToFileLevelDatasetMapper.prepare_data(line_level_dataset, False)
    data = {
        'text': docs,
        'label': [1 if label == True else 0 for label in labels]
    }
    train_df = pd.DataFrame(data)

    max_features = 200000
    max_senten_len = 40
    max_senten_num = 6
    embed_size = 100
    VALIDATION_SPLIT = 0.2
    token_extractor = CustomTokenExtractor()

    texts_sentences = []
    labels = []
    texts = []
    sent_lens = []
    sent_nums = []
    for idx in range(train_df.text.shape[0]):
        # TODO: clean text
        text = train_df.text[idx]
        texts.append(text)
        sentences = [token_extractor.extract_tokens(line) for line in text.splitlines()]
        sent_nums.append(len(sentences))
        for sent in sentences:
            sent_lens.append(len(sent))
        texts_sentences.append(sentences)

    sns.histplot(sent_lens, bins=200)
    plt.show()

    sns.histplot(sent_nums)
    plt.show()

    keras_tokenizer = KerasTokenizer.train(texts, metadata={
        'num_words': max_features,
        'oov_token': '<oov>',
        'to_lowercase': False,
        'token_extractor': token_extractor
    })
    tokenizer = keras_tokenizer.tokenizer

    data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
    for text_idx, sentences in enumerate(texts_sentences):
        for sent_idx, sent in enumerate(sentences):
            if sent_idx < max_senten_num:
                wordTokens = sent
                k = 0
                for _, word in enumerate(wordTokens):
                    try:
                        if k < max_senten_len and tokenizer.word_index[word] < max_features:
                            data[text_idx, sent_idx, k] = tokenizer.word_index[word]
                            k = k + 1
                    except:
                        print(word)
                        pass

    print(data.shape)

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = train_df['label']

    print('Shape of data tensor:', data.shape)
    print('Shape of labels tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels.iloc[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    print('Number of positive and negative reviews in training and validation set')
    # print(y_train.columns.tolist())
    print(y_train.sum(axis=0).tolist())
    print(y_val.sum(axis=0).tolist())

    REG_PARAM = 1e-13
    l2_reg = regularizers.l2(REG_PARAM)

    word2vec = GensimWord2VecModel.train(texts, metadata={
        'embedding_dim': embed_size,
        'dataset_name': "test1",
        'token_extractor': token_extractor
    })
    embedding_matrix = word2vec.get_embedding_matrix(tokenizer.word_index, keras_tokenizer.get_vocab_size(),
                                                     embed_size)

    embedding_layer = Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix],
                                input_length=max_senten_len, trainable=False)

    word_input = Input(shape=(max_senten_len,), dtype='float32')
    word_sequences = embedding_layer(word_input)
    word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
    word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
    word_att = AttentionWithContext()(word_dense)
    wordEncoder = Model(word_input, word_att)
    sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
    sent_encoder = TimeDistributed(wordEncoder)(sent_input)
    sent_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
    sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
    sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
    preds = Dense(1, activation='sigmoid')(sent_att)
    model = Model(sent_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('best_model.h5', verbose=0, monitor='val_loss', save_best_only=True, mode='auto')

    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=512,
                        callbacks=[checkpoint])

    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
