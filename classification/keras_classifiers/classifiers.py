import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, OneSidedSelection, CondensedNearestNeighbour, \
    NearMiss
from imblearn.combine import SMOTETomek
from keras import layers, Sequential
from keras.src import regularizers
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.optimizers import Adam
from keras.src.utils import pad_sequences, plot_model
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader

from classification.keras_classifiers.Rept import ReptileDataset
from classification.keras_classifiers.attention_with_context import AttentionWithContext
from classification.keras_classifiers.l2l import BinaryClassifier, FewShotTextDataset
from classification.models import ClassifierModel, ClassificationDataset
from config import KERAS_SAVE_PREDICTION_DIR, SIMPLE_KERAS_PREDICTION_DIR, KERAS_CNN_SAVE_PREDICTION_DIR
import learn2learn as l2l
import torch
from torch import nn, optim
import random

import torch
from torch.cuda import device


class KerasClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        raise NotImplementedError()

    @classmethod
    def train(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        dataset_name = metadata.get('dataset_name')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        imbalanced_learn_method = metadata.get(
            'imbalanced_learn_method')  # smote, adasyn, rus, tomek, nearmiss, smotetomek

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        X_train, Y_train = cls.get_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

        minority_class_count = np.sum(Y_train == 1)
        majority_class_count = np.sum(Y_train == 0)
        print(f'minority_class_count: {minority_class_count}')
        print(f'majority_class_count: {majority_class_count}')
        desired_majority_count = minority_class_count * 2
        sampling_strategy = {0: desired_majority_count, 1: minority_class_count}

        print(f'imbalanced_learn_method = {imbalanced_learn_method}')
        if imbalanced_learn_method == 'smote':
            sm = SMOTE(random_state=42)
            X_train, Y_train = sm.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_train, Y_train = adasyn.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'rus':
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, Y_train = rus.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'tomek':
            tomek = TomekLinks()
            X_train, Y_train = tomek.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'nearmiss':
            near_miss = NearMiss()
            X_train, Y_train = near_miss.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'smotetomek':
            smotetomek = SMOTETomek(random_state=42)
            X_train, Y_train = smotetomek.fit_resample(X_train, Y_train)

        if metadata.get('perform_k_fold_cross_validation'):
            cls.k_fold_cross_validation(X_train, Y_train, batch_size, embedding_dim, embedding_matrix, epochs,
                                        max_seq_len, vocab_size)

        model = cls.build_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            max_seq_len=max_seq_len,
            show_summary=True
        )

        if class_weight_strategy == 'up_weight_majority':
            if imbalanced_learn_method not in {'nearmiss', 'rus', 'tomek'}:
                raise Exception(f"imbalanced_learn_method {imbalanced_learn_method} is not a down-sampling so "
                                f"majority up-weighing is not allowed")
            class_weight_dict = {0: majority_class_count / desired_majority_count, 1: 1}
        elif class_weight_strategy == 'up_weight_minority':
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
            class_weight_dict = dict(enumerate(class_weights))
        else:
            class_weight_dict = None
        print(f'class_weight_dict = {class_weight_dict}')

        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        model_checkpoint = ModelCheckpoint(filepath=f'models/{cls.__name__}-{dataset_name}.h5', monitor='val_loss',
                                           save_best_only=True)

        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            validation_data=cls.get_X_and_Y(validation_dataset, embedding_model, max_seq_len),
            callbacks=[early_stopping, model_checkpoint]
        )
        cls.plot_history(history)
        loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        return cls(model, embedding_model)

    @classmethod
    def k_fold_cross_validation(cls, X, Y, batch_size, embedding_dim, embedding_matrix, epochs, max_seq_len,
                                vocab_size):
        kf = KFold(n_splits=10)
        validation_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            model = cls.build_model(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                embedding_matrix=embedding_matrix,
                max_seq_len=max_seq_len,
                show_summary=False
            )

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
            class_weight_dict = dict(enumerate(class_weights))
            print(f'class_weight_dict = {class_weight_dict}')

            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0,
                class_weight=class_weight_dict
            )
            validation_score = model.evaluate(X_test, y_test)
            validation_scores.append(validation_score)
        # get average score
        validation_score = np.average(validation_scores)
        print(f'validation_score = {validation_score}')

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = dataset.get_texts(), dataset.get_labels()

        X_test = self.embedding_model.text_to_indexes(codes)

        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)
        return self.model.predict(X_test)

    @classmethod
    def get_X_and_Y(cls, classification_dataset, embedding_model, max_seq_len):
        codes, labels = classification_dataset.get_texts(), classification_dataset.get_labels()

        X = embedding_model.text_to_indexes(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        Y = np.array([1 if label == True else 0 for label in labels])
        return X, Y

    @classmethod
    def export_model_plot(cls, model):
        Path('./plots').mkdir(parents=True, exist_ok=True)
        plot_model(model, to_file=f'plots/{cls.__name__}.pdf', show_shapes=False, show_layer_names=True)


class KerasDenseClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()
        model.add(layers.Dense(512, input_dim=embedding_dim, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        cls.export_model_plot(model)
        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(SIMPLE_KERAS_PREDICTION_DIR, dataset_name + '.csv')


class KerasDenseClassifierWithEmbedding(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()
        model.add(
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_seq_len,
                trainable=True,
                mask_zero=True
            )
        )
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        cls.export_model_plot(model)
        return model


class KerasDenseClassifierWithExternalEmbedding(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()
        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True,
                mask_zero=True
            )
        )
        # model.add(layers.GlobalMaxPool1D())
        # model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dropout(0.25))
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dropout(0.25))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dropout(0.25))
        # model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dropout(0.25))
        # model.add(layers.MaxPooling1D())
        # model.add(layers.Dense(10, activation='relu'))
        # Project onto a single unit output layer, and squash it with a sigmoid:
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        cls.export_model_plot(model)
        return model


class KerasCNNClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        inputs = tf.keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, input_length=max_seq_len)(inputs)
        # x = layers.Conv1D(100, 4, padding="same", activation="relu")(x)
        x = layers.Conv1D(100, 5, padding="same", activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Dense(100, activation="relu")(x)
        # x = layers.Dropout(0.25)(x)
        # x = layers.Dense(20, activation="relu")(x)
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
        model = tf.keras.Model(inputs, predictions)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        if kwargs.get('show_summary'):
            model.summary()

        cls.export_model_plot(model)
        return model

    # @classmethod
    # def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
    #     model = Sequential()
    #     word_embeddings = layers.Embedding(vocab_size, embedding_dim, input_length=max_seq_len, mask_zero=True)
    #     model.add(word_embeddings)
    #     # Add a dimension at index 1
    #     model.add(layers.Reshape((max_seq_len, embedding_dim, 1)))
    #     model.add(layers.Conv2D(100, (5, embedding_dim), padding="valid", activation="relu"))
    #     # Squeeze operation (remove a dimension)
    #     model.add(layers.Lambda(lambda x: tf.squeeze(x, axis=-2)))
    #     model.add(layers.MaxPooling1D(pool_size=model.output_shape[1]))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dropout(0.25))
    #     model.add(layers.Dense(1, activation='sigmoid', name="predictions"))
    #     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #     if kwargs.get('show_summary'):
    #         model.summary()
    #     return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasCNNClassifierWithEmbedding(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()

        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True,
                mask_zero=True
            )
        )

        model.add(layers.Conv1D(100, 5, padding="same", activation="relu"))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(100, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        # cls.export_model_plot(model)
        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_CNN_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasLSTMClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        inputs = tf.keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], mask_zero=True)(inputs)
        x = layers.LSTM(64)(x)
        x = layers.Dense(64, name='FC1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1, name='out_layer')(x)
        x = layers.Activation('sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        cls.export_model_plot(model)
        return model


class KerasBiLSTMClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()

        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True,
                mask_zero=True
            )
        )

        # Define parameter
        n_lstm = 32
        drop_lstm = 0.2
        model.add(layers.Bidirectional(layers.LSTM(n_lstm, return_sequences=False)))
        model.add(layers.Dense(64, name='FC1'))
        model.add(layers.Dropout(drop_lstm))

        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        cls.export_model_plot(model)
        return model

    # @classmethod
    # def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
    #     model = Sequential()
    #     model.add(
    #         layers.Embedding(
    #             vocab_size,
    #             embedding_dim,
    #             weights=[embedding_matrix],
    #             input_length=max_seq_len,
    #             trainable=True,
    #             mask_zero=True
    #         )
    #     )
    #     model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    #     model.add(layers.Bidirectional(layers.LSTM(32)))
    #     model.add(layers.Dropout(0.5))
    #     model.add(layers.Dense(64, activation='relu'))
    #     model.add(layers.Dense(1, activation='sigmoid'))
    #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     return model


class KerasGRUClassifier(KerasClassifier):
    # @classmethod
    # def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
    #     model = Sequential()
    #
    #     model.add(
    #         layers.Embedding(
    #             vocab_size, embedding_dim,
    #             weights=[embedding_matrix],
    #             input_length=max_seq_len,
    #             trainable=True,
    #             mask_zero=True
    #         )
    #     )
    #
    #     # model.add(layers.SpatialDropout1D(0.2))
    #     model.add(layers.GRU(64, return_sequences=False))
    #     model.add(layers.Dropout(0.2))
    #     model.add(layers.Dense(64, activation='relu'))
    #     model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
    #
    #     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #     model.summary()
    #     return model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()
        model.add(
            layers.Embedding(
                vocab_size,
                embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True,
                mask_zero=True
            )
        )

        model.add(layers.SpatialDropout1D(0.2))

        model.add(layers.Bidirectional(layers.GRU(64, return_sequences=True)))
        model.add(layers.Bidirectional(layers.GRU(32, return_sequences=False)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        cls.export_model_plot(model)
        return model


class KerasCNNandLSTMClassifier(KerasClassifier):
    # @classmethod
    # def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
    #     model = Sequential()
    #
    #     model.add(
    #         layers.Embedding(
    #             vocab_size, embedding_dim,
    #             weights=[embedding_matrix],
    #             input_length=max_seq_len,
    #             trainable=True,
    #             mask_zero=True
    #         )
    #     )
    #
    #     model.add(layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    #     model.add(layers.MaxPooling1D(pool_size=2))
    #     model.add(layers.Dropout(0.25))
    #     model.add(layers.Bidirectional(layers.GRU(16, return_sequences=False)))
    #     model.add(layers.Dense(32, activation="relu"))
    #     model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
    #
    #     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    #     model.summary()
    #     return model
    #
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()
        model.add(
            layers.Embedding(
                vocab_size,
                embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True,
                mask_zero=True
            )
        )
        model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Bidirectional(layers.GRU(64)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        cls.export_model_plot(model)
        return model


class KerasHANClassifier(KerasClassifier):
    max_sent_num = 100

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        learning_rate = kwargs.get('learning_rate', 0.001)
        dropout_ratio = kwargs.get('dropout_ratio', 0.5)
        print(f"learning_rate = {learning_rate}")
        print(f"dropout_ratio = {dropout_ratio}")
        token_encoder_bi_gru_hidden_cells_count = 32
        token_attention_mlp_hidden_cells_count = 64
        line_encoder_bi_gru_hidden_cells_count = 32
        line_attention_mlp_hidden_cells_count = 64

        REG_PARAM = 1e-13
        l2_reg = regularizers.l2(REG_PARAM)
        l2_reg = None
        optimizer = Adam(learning_rate=learning_rate)
        embedding_layer = layers.Embedding(
            vocab_size, embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_len,
            trainable=True,
            mask_zero=True
        )

        # Token level attention model
        token_input = layers.Input(shape=(max_seq_len,), dtype='float32')
        token_sequences = embedding_layer(token_input)
        token_gru = layers.Bidirectional(
            layers.GRU(
                units=token_encoder_bi_gru_hidden_cells_count,
                return_sequences=True,
                kernel_regularizer=l2_reg
            )
        )(token_sequences)
        token_gru = layers.TimeDistributed(layers.LayerNormalization())(token_gru)
        # token_gru = layers.LayerNormalization()(token_gru)
        token_dense = layers.TimeDistributed(
            layers.Dense(token_attention_mlp_hidden_cells_count, kernel_regularizer=l2_reg))(token_gru)
        token_attention = AttentionWithContext()(token_dense)
        token_attention_model = tf.keras.Model(token_input, token_attention)

        # Line level attention model
        line_input = layers.Input(shape=(KerasHANClassifier.max_sent_num, max_seq_len), dtype='float32')
        line_encoder = layers.TimeDistributed(token_attention_model)(line_input)
        line_gru = layers.Bidirectional(
            layers.GRU(
                units=line_encoder_bi_gru_hidden_cells_count,
                return_sequences=True,
                kernel_regularizer=l2_reg)
        )(line_encoder)
        line_gru = layers.TimeDistributed(layers.LayerNormalization())(line_gru)
        # line_gru = layers.LayerNormalization()(line_gru)
        line_dense = layers.TimeDistributed(
            layers.Dense(line_attention_mlp_hidden_cells_count, kernel_regularizer=l2_reg))(line_gru)
        line_attention = layers.Dropout(dropout_ratio)(AttentionWithContext()(line_dense))
        preds = layers.Dense(1, activation='sigmoid')(line_attention)
        model = tf.keras.Model(line_input, preds)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        cls.export_model_plot(model)
        return model

    @classmethod
    def train(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        learning_rate = metadata.get('learning_rate')
        dropout_ratio = metadata.get('dropout_ratio')

        if max_seq_len is None:
            raise Exception("max_seq_len can not be none in this model")

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        codes, labels = train_dataset.get_texts(), train_dataset.get_labels()

        codes_3d = np.zeros((len(codes), KerasHANClassifier.max_sent_num, max_seq_len), dtype='int32')
        for file_idx, file_code in enumerate(codes):
            for line_idx, line in enumerate(file_code.splitlines()):
                if line_idx >= KerasHANClassifier.max_sent_num:
                    continue

                X = embedding_model.text_to_indexes([line])[0]
                X = pad_sequences([X], padding='post', maxlen=max_seq_len)
                codes_3d[file_idx, line_idx, :] = X

        Y = np.array([1 if label == True else 0 for label in labels])

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
        class_weight_dict = dict(enumerate(class_weights))
        print(f'class_weight_dict = {class_weight_dict}')

        model = cls.build_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            max_seq_len=max_seq_len,
            learning_rate=learning_rate,
            dropout_ratio=dropout_ratio
        )
        history = model.fit(
            codes_3d, Y,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict
        )
        cls.plot_history(history)
        loss, accuracy = model.evaluate(codes_3d, Y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        return cls(model, embedding_model)

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = dataset.get_texts(), dataset.get_labels()

        codes_3d = np.zeros((len(codes), KerasHANClassifier.max_sent_num, max_seq_len), dtype='int32')
        for file_idx, file_code in enumerate(codes):
            for line_idx, line in enumerate(file_code.splitlines()):
                if line_idx >= KerasHANClassifier.max_sent_num:
                    continue

                X = self.embedding_model.text_to_indexes([line])[0]
                X = pad_sequences([X], padding='post', maxlen=max_seq_len)
                codes_3d[file_idx, line_idx, :] = X

        return self.model.predict(codes_3d)


class BalancedBaggingUnderSampleClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def train(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
              metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        dataset_name = metadata.get('dataset_name')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        imbalanced_learn_method = metadata.get(
            'imbalanced_learn_method')  # smote, adasyn, rus, tomek, nearmiss, smotetomek

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        X_train, Y_train = cls.get_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

        minority_class_count = np.sum(Y_train == 1)
        majority_class_count = np.sum(Y_train == 0)
        print(f'minority_class_count: {minority_class_count}')
        print(f'majority_class_count: {majority_class_count}')
        desired_majority_count = minority_class_count * 2
        sampling_strategy = {0: desired_majority_count, 1: minority_class_count}

        from imblearn.ensemble import BalancedBaggingClassifier
        from imblearn.under_sampling import RandomUnderSampler

        # Exactly Balanced Bagging
        ebb = BalancedBaggingClassifier(sampler=RandomUnderSampler())
        cv_results = cross_validate(ebb, X_train, Y_train, scoring="balanced_accuracy")

        print(f"{cv_results['test_score'].mean():.3f} +/- {cv_results['test_score'].std():.3f}")
        ebb.fit(X_train, Y_train)
        return cls(ebb, embedding_model)

    def predict(self, dataset: ClassificationDataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = dataset.get_texts(), dataset.get_labels()

        X_test = self.embedding_model.text_to_indexes(codes)

        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)
        return [[x[self.model.classes_.tolist().index(1)]] for x in self.model.predict_proba(X_test)]

    @classmethod
    def get_X_and_Y(cls, classification_dataset, embedding_model, max_seq_len):
        codes, labels = classification_dataset.get_texts(), classification_dataset.get_labels()

        X = embedding_model.text_to_indexes(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        Y = np.array([1 if label == True else 0 for label in labels])
        return X, Y


class EnsembleClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model_input = kwargs.get('model_input')

        embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_len,
            trainable=True,
            mask_zero=True
        )(model_input)

        conv_layer = layers.Conv1D(100, 5, padding="same", activation="relu")(embedding_layer)
        global_max_pooling = layers.GlobalMaxPooling1D()(conv_layer)
        dropout_layer = layers.Dropout(0.25)(global_max_pooling)
        dense_layer = layers.Dense(100, activation="relu")(dropout_layer)
        predictions = layers.Dense(1, activation="sigmoid")(dense_layer)

        model = tf.keras.Model(inputs=model_input, outputs=predictions)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model

    @classmethod
    def train(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
              metadata=None):
        np.random.seed(42)
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        dataset_name = metadata.get('dataset_name')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        imbalanced_learn_method = metadata.get(
            'imbalanced_learn_method')  # smote, adasyn, rus, tomek, nearmiss, smotetomek

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        X_train, Y_train = cls.get_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

        minority_class_count = np.sum(Y_train == 1)
        majority_class_count = np.sum(Y_train == 0)
        print(f'minority_class_count: {minority_class_count}')
        print(f'majority_class_count: {majority_class_count}')

        splits_count = majority_class_count // minority_class_count
        # splits_count = 3

        X_train_minority_samples = X_train[Y_train == 1]
        X_train_majority_samples = X_train[Y_train == 0]

        sub_models = []
        model_input = layers.Input(shape=(max_seq_len,))

        for majority_split in np.array_split(X_train_majority_samples, 1):
            X_train_balanced = np.vstack((X_train_minority_samples, majority_split))
            Y_train_balanced = np.hstack((np.ones(len(X_train_minority_samples)), np.zeros(len(majority_split))))

            combined_data = list(zip(X_train_balanced, Y_train_balanced))
            np.random.shuffle(combined_data)

            X_train_shuffled, Y_train_shuffled = zip(*combined_data)

            X_train_shuffled = np.array(X_train_shuffled)
            Y_train_shuffled = np.array(Y_train_shuffled)

            sub_model = cls.build_model(vocab_size, embedding_dim, embedding_matrix,
                                        max_seq_len, model_input=model_input)
            history = sub_model.fit(
                X_train_shuffled, Y_train_shuffled,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=cls.get_X_and_Y(validation_dataset, embedding_model, max_seq_len),
            )
            cls.plot_history(history)
            loss, accuracy = sub_model.evaluate(X_train, Y_train, verbose=False)
            print("Training Accuracy: {:.4f}".format(accuracy))
            sub_models.append(sub_model)

        outputs = [model.outputs[0] for model in sub_models]
        y = layers.Average()(outputs)
        model = tf.keras.Model(model_input, y, name='ensemble')
        KerasClassifier.export_model_plot(model)

        # outputs = layers.Concatenate()(outputs)
        # x = layers.Dense(50, activation="relu")(outputs)
        # predictions = layers.Dense(1, activation="sigmoid")(x)
        # model = tf.keras.Model(model_input, predictions)
        # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        #
        # history = model.fit(
        #     X_train, Y_train,
        #     epochs=epochs,
        #     batch_size=batch_size,
        #     validation_data=cls.get_X_and_Y(validation_dataset, embedding_model, max_seq_len),
        # )
        # cls.plot_history(history)
        # loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
        # print("Training Accuracy: {:.4f}".format(accuracy))

        return cls(model, embedding_model)

    def predict(self, dataset: ClassificationDataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = dataset.get_texts(), dataset.get_labels()

        X_test = self.embedding_model.text_to_indexes(codes)

        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)
        return self.model.predict(X_test)

    @classmethod
    def get_X_and_Y(cls, classification_dataset, embedding_model, max_seq_len):
        codes, labels = classification_dataset.get_texts(), classification_dataset.get_labels()

        X = embedding_model.text_to_indexes(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        Y = np.array([1 if label == True else 0 for label in labels])
        return X, Y


class L2LClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

class ReptileClassifier(KerasClassifier):
    @classmethod
    def train(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        dataset_name = metadata.get('dataset_name')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        imbalanced_learn_method = metadata.get(
            'imbalanced_learn_method')  # smote, adasyn, rus, tomek, nearmiss, smotetomek

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        X_train, Y_train = cls.get_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

        minority_class_count = np.sum(Y_train == 1)
        majority_class_count = np.sum(Y_train == 0)
        print(f'minority_class_count: {minority_class_count}')
        print(f'majority_class_count: {majority_class_count}')
        desired_majority_count = minority_class_count * 2
        sampling_strategy = {0: desired_majority_count, 1: minority_class_count}

        print(f'imbalanced_learn_method = {imbalanced_learn_method}')
        if imbalanced_learn_method == 'smote':
            sm = SMOTE(random_state=42)
            X_train, Y_train = sm.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_train, Y_train = adasyn.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'rus':
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, Y_train = rus.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'tomek':
            tomek = TomekLinks()
            X_train, Y_train = tomek.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'nearmiss':
            near_miss = NearMiss()
            X_train, Y_train = near_miss.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'smotetomek':
            smotetomek = SMOTETomek(random_state=42)
            X_train, Y_train = smotetomek.fit_resample(X_train, Y_train)

        if metadata.get('perform_k_fold_cross_validation'):
            cls.k_fold_cross_validation(X_train, Y_train, batch_size, embedding_dim, embedding_matrix, epochs,
                                        max_seq_len, vocab_size)

        X_val, Y_val = cls.get_X_and_Y(validation_dataset, embedding_model, max_seq_len)
        learning_rate = 0.003
        meta_step_size = 0.25

        inner_batch_size = 25
        eval_batch_size = 25

        meta_iters = 2000
        eval_iters = 5
        inner_iters = 4

        eval_interval = 1
        train_shots = 20
        shots = 5
        classes = 2

        train_reptile_dataset = ReptileDataset(X_train, Y_train)
        validation_reptile_dataset = ReptileDataset(X_val, Y_val)

        model = KerasCNNClassifierWithEmbedding.build_model(vocab_size, embedding_dim, embedding_matrix, max_seq_len)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        training = []
        testing = []
        for meta_iter in range(meta_iters):
            frac_done = meta_iter / meta_iters
            cur_meta_step_size = (1 - frac_done) * meta_step_size
            # Temporarily save the weights from the model.
            old_vars = model.get_weights()
            # Get a sample from the full dataset.
            mini_dataset = train_reptile_dataset.get_mini_dataset(
                inner_batch_size, inner_iters, train_shots
            )
            for embeddings, labels in mini_dataset:
                with tf.GradientTape() as tape:
                    preds = model(embeddings)
                    loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(labels, axis=-1), preds)
                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            new_vars = model.get_weights()
            # Perform SGD for the meta step.
            for var in range(len(new_vars)):
                new_vars[var] = old_vars[var] + (
                        (new_vars[var] - old_vars[var]) * cur_meta_step_size
                )
            # After the meta-learning step, reload the newly-trained weights into the model.
            model.set_weights(new_vars)
            # Evaluation loop
            if meta_iter % eval_interval == 0:
                accuracies = []
                for reptile_dataset in (train_reptile_dataset, validation_reptile_dataset):
                    # Sample a mini dataset from the full dataset.
                    train_set, test_embeddings, test_labels = reptile_dataset.get_mini_dataset(
                        eval_batch_size, eval_iters, shots, split=True
                    )
                    old_vars = model.get_weights()
                    # Train on the samples and get the resulting accuracies.
                    for embeddings, labels in train_set:
                        with tf.GradientTape() as tape:
                            preds = model(embeddings)
                            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(labels, axis=-1), preds)
                        grads = tape.gradient(loss, model.trainable_weights)
                        optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    test_preds = model.predict(test_embeddings, verbose=0).flatten()
                    test_preds = np.round(test_preds)
                    num_correct = (test_preds == test_labels).sum()
                    # Reset the weights after getting the evaluation accuracies.
                    model.set_weights(old_vars)
                    accuracies.append(num_correct / classes)
                training.append(accuracies[0])
                testing.append(accuracies[1])
                if meta_iter % 100 == 0:
                    print(
                        "batch %d: train=%f test=%f" % (meta_iter, accuracies[0], accuracies[1])
                    )

        # First, some preprocessing to smooth the training and testing arrays for display.
        window_length = 100
        train_s = np.r_[
            training[window_length - 1: 0: -1],
            training,
            training[-1:-window_length:-1],
        ]
        test_s = np.r_[
            testing[window_length - 1: 0: -1], testing, testing[-1:-window_length:-1]
        ]
        w = np.hamming(window_length)
        train_y = np.convolve(w / w.sum(), train_s, mode="valid")
        test_y = np.convolve(w / w.sum(), test_s, mode="valid")

        # Display the training accuracies.
        x = np.arange(0, len(test_y), 1)
        plt.plot(x, test_y, x, train_y)
        plt.legend(["test", "train"])
        plt.grid()

        train_set, test_embeddings, test_labels = validation_reptile_dataset.get_mini_dataset(
            eval_batch_size, eval_iters, shots, split=True
        )
        for images, labels in train_set:
            with tf.GradientTape() as tape:
                preds = model(images)
                loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(labels, axis=-1), preds)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        test_preds = model.predict(test_embeddings)
        test_preds = tf.argmax(test_preds).numpy()


        plt.show()
        return cls(model, embedding_model)

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        X_test, Y_test = self.get_X_and_Y(dataset, self.embedding_model, max_seq_len)

        test_reptile_dataset = ReptileDataset(X_test, Y_test)
        learning_rate = 0.003
        meta_step_size = 0.25

        inner_batch_size = 25
        eval_batch_size = 25

        meta_iters = 200
        eval_iters = 5
        inner_iters = 4

        eval_interval = 1
        train_shots = 20
        shots = 5
        classes = 2
        train_set, test_images, test_labels = test_reptile_dataset.get_mini_dataset(
            eval_batch_size, eval_iters, shots, split=True
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        for images, labels in train_set:
            with tf.GradientTape() as tape:
                preds = self.model(images)
                loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(labels, axis=-1), preds)
            grads = tape.gradient(loss, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        test_preds = self.model.predict(test_images)
        test_preds = np.round(test_preds)
        print(test_labels)
        print(test_preds)

        return self.model.predict(X_test)


