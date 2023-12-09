import tensorflow as tf
import numpy as np
from keras.layers import Dense, Layer, GRU, Dropout, LayerNormalization, Embedding
from keras.src.layers import Bidirectional, Attention, MultiHeadAttention


class HierarchicalAttentionNetwork(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim, word_gru_num_layers,
                 sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(HierarchicalAttentionNetwork, self).__init__()

        self.sent_attention = SentenceAttention(vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                                                word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim,
                                                use_layer_norm, dropout)

        self.fc = Dense(1)
        self.sig = tf.keras.layers.Activation('sigmoid')

        self.use_layer_norm = use_layer_norm
        self.dropout = dropout

    def call(self, code_tensor):
        code_lengths = []
        sent_lengths = []

        for file in code_tensor:
            code_line = []
            code_lengths.append(len(file))
            for line in file:
                code_line.append(len(line))
            sent_lengths.append(code_line)

        code_tensor = tf.convert_to_tensor(code_tensor, dtype=tf.int32)
        code_lengths = tf.convert_to_tensor(code_lengths, dtype=tf.int32)
        sent_lengths = tf.convert_to_tensor(sent_lengths, dtype=tf.int32)

        code_embeds, word_att_weights, sent_att_weights, sents = self.sent_attention(code_tensor, code_lengths,
                                                                                     sent_lengths)

        scores = self.fc(code_embeds)
        final_scores = self.sig(scores)

        return final_scores, word_att_weights, sent_att_weights, sents


class SentenceAttention(Layer):
    def __init__(self, vocab_size, embed_dim, word_gru_hidden_dim, sent_gru_hidden_dim,
                 word_gru_num_layers, sent_gru_num_layers, word_att_dim, sent_att_dim, use_layer_norm, dropout):
        super(SentenceAttention, self).__init__()

        self.word_attention = WordAttention(vocab_size, embed_dim, word_gru_hidden_dim, word_gru_num_layers,
                                            word_att_dim, use_layer_norm, dropout)

        self.gru = Bidirectional(GRU(sent_gru_hidden_dim, return_sequences=True, return_state=True,
                                     dropout=dropout))
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

        self.sent_attention = Dense(sent_att_dim, activation='tanh')

        self.sentence_context_vector = Dense(1, use_bias=False)

    def call(self, code_tensor, code_lengths, sent_lengths):
        code_lengths, code_perm_idx = tf.sort(code_lengths, direction='DESCENDING')
        code_tensor = tf.gather(code_tensor, code_perm_idx)
        sent_lengths = tf.gather(sent_lengths, code_perm_idx)

        packed_sents = tf.RaggedTensor.from_tensor(code_tensor).to_tensor()

        valid_bsz = tf.sort(tf.math.segment_sum(tf.ones_like(packed_sents[:, 0]), code_lengths), direction='DESCENDING')

        packed_sent_lengths = tf.RaggedTensor.from_tensor(sent_lengths).to_tensor()

        sents, word_att_weights = self.word_attention(packed_sents, packed_sent_lengths)

        sents = self.dropout(sents)

        packed_sents = self.gru(sents)

        if self.use_layer_norm:
            normed_sents = self.layer_norm(packed_sents)
        else:
            normed_sents = packed_sents

        att = self.sent_attention(normed_sents)
        att = self.sentence_context_vector(att)

        val = tf.reduce_max(att)
        att = tf.exp(att - val)

        att = tf.RaggedTensor.from_tensor(att).to_tensor()
        sent_att_weights = att / tf.reduce_sum(att, axis=1, keepdims=True)

        code_tensor = tf.RaggedTensor.from_tensor(code_tensor).to_tensor()

        code_tensor = code_tensor * tf.expand_dims(sent_att_weights, axis=2)
        code_tensor = tf.reduce_sum(code_tensor, axis=1)

        word_att_weights = tf.RaggedTensor.from_tensor(word_att_weights).to_tensor()

        _, code_tensor_unperm_idx = tf.argsort(code_perm_idx)
        code_tensor = tf.gather(code_tensor, code_tensor_unperm_idx)

        word_att_weights = tf.gather(word_att_weights, code_tensor_unperm_idx)
        sent_att_weights = tf.gather(sent_att_weights, code_tensor_unperm_idx)

        return code_tensor, word_att_weights, sent_att_weights, sents


class WordAttention(Layer):
    def __init__(self, vocab_size, embed_dim, gru_hidden_dim, gru_num_layers, att_dim, use_layer_norm, dropout):
        super(WordAttention, self).__init__()

        self.embeddings = Embedding(vocab_size, embed_dim)

        self.gru = GRU(gru_hidden_dim, return_sequences=True, return_state=True, dropout=dropout)
        Attention()
        MultiHeadAttention()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

        self.attention = Dense(att_dim, activation='tanh')

        self.context_vector = Dense(1, use_bias=False)

    def call(self, sents, sent_lengths):
        sent_lengths, sent_perm_idx = tf.sort(sent_lengths, direction='DESCENDING')
        sents = tf.gather(sents, sent_perm_idx)

        sents = self.embeddings(sents)

        packed_words = tf.RaggedTensor.from_tensor(sents).to_tensor()

        valid_bsz = tf.sort(tf.math.segment_sum(tf.ones_like(packed_words[:, 0]), sent_lengths), direction='DESCENDING')

        packed_words, state = self.gru(packed_words)

        if self.use_layer_norm:
            normed_words = self.layer_norm(packed_words)
        else:
            normed_words = packed_words

        att = self.attention(normed_words)
        att = self.context_vector(att)

        val = tf.reduce_max(att)
        att = tf.exp(att - val)

        att = tf.RaggedTensor.from_tensor(att).to_tensor()
        att_weights = att / tf.reduce_sum(att, axis=1, keepdims=True)

        sents = tf.RaggedTensor.from_tensor(sents).to_tensor()

        sents = sents * tf.expand_dims(att_weights, axis=2)
        sents = tf.reduce_sum(sents, axis=1)

        _, sent_unperm_idx = tf.argsort(sent_perm_idx)
        sents = tf.gather(sents, sent_unperm_idx)

        att_weights = tf.gather(att_weights, sent_unperm_idx)

        return sents, att_weights
