import os

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import layers, Sequential
from keras.src.utils import pad_sequences
from sklearn.model_selection import KFold

from classification.keras_classifiers.attention_with_context import AttentionWithContext
from classification.models import ClassifierModel
from config import KERAS_SAVE_PREDICTION_DIR, SIMPLE_KERAS_PREDICTION_DIR, KERAS_CNN_SAVE_PREDICTION_DIR


class KerasClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        raise NotImplementedError()

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        codes, labels = df['text'], df['label']

        X = embedding_model.text_to_indexes(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        if max_seq_len is None:
            max_seq_len = X.shape[1]

        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(X, Y)

        cls.k_fold_cross_validation(X, Y, batch_size, embedding_dim, embedding_matrix, epochs, max_seq_len, vocab_size)

        model = cls.build_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            max_seq_len=max_seq_len
        )

        history = model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
        )
        cls.plot_history(history)
        loss, accuracy = model.evaluate(X, Y, verbose=False)
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
                max_seq_len=max_seq_len
            )
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test)

            )
            validation_score = model.evaluate(X_test, y_test)
            validation_scores.append(validation_score)
        # get average score
        validation_score = np.average(validation_scores)
        print(f'validation_score = {validation_score}')

    def predict(self, df, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = df['text'], df['label']

        X_test = self.embedding_model.text_to_indexes(codes)

        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)
        return self.model.predict(X_test)


class KerasDenseClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()
        model.add(layers.Dense(512, input_dim=embedding_dim, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(SIMPLE_KERAS_PREDICTION_DIR, dataset_name + '.csv')


class KerasDenseClassifierWithEmbedding(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()
        model.add(
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_seq_len,
                trainable=True
            )
        )
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model


class KerasDenseClassifierWithExternalEmbedding(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()
        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True
            )
        )
        # model.add(layers.GlobalMaxPool1D())
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.25))
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dropout(0.25))
        # model.add(layers.Dense(128, activation='relu'))
        # model.add(layers.Dropout(0.25))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dropout(0.25))
        # model.add(layers.MaxPooling1D())
        # model.add(layers.Dense(10, activation='relu'))
        # Project onto a single unit output layer, and squash it with a sigmoid:
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()

        return model


class KerasCNNClassifierWithEmbedding(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()

        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                # weights=[embedding_matrix] if embedding_matrix is not None else None,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True
            )
        )

        # model.add(layers.Bidirectional(layers.GRU(64, return_sequences=True)))
        # model.add(layers.Bidirectional(layers.GRU(64)))

        # model.add(layers.Dropout(0.2))

        # Modified CNN layers similar to the provided architecture.
        # model.add(layers.Conv1D(100, 5, padding="same", activation="relu", strides=1))
        model.add(layers.Conv1D(100, 5, padding="same", activation="relu"))
        # model.add(layers.MaxPooling1D())
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Flatten())
        # model.add(layers.Dropout(0.2))

        # model.add(layers.Dense(8, activation="relu"))
        # model.add(layers.Flatten())
        # model.add(layers.Dropout(0.2))

        # model.add(layers.Dense(1024, activation="relu"))
        # model.add(layers.Dropout(0.2))

        # Add another Conv1D layer for complexity
        # model.add(layers.Conv1D(100, 5, padding="same", activation="relu"))
        # model.add(layers.MaxPooling1D())
        # model.add(layers.Dropout(0.2))

        model.add(layers.GlobalMaxPool1D())
        # model.add(layers.Dropout(0.1))
        # Vanilla hidden layer:
        # model.add(layers.Conv1D(filters=4, kernel_size=5, padding='same', activation='relu'))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(16, activation="relu"))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.GRU(2))
        # Project onto a single unit output layer, and squash it with a sigmoid:
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_CNN_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasCNNClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        inputs = tf.keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(vocab_size, embedding_dim)(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
        model = tf.keras.Model(inputs, predictions)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasLSTMClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        inputs = tf.keras.Input(shape=(None,), dtype="int64")
        x = layers.Embedding(vocab_size, embedding_dim)(inputs)
        x = layers.LSTM(64)(x)
        x = layers.Dense(256, name='FC1')(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, name='out_layer')(x)
        x = layers.Activation('sigmoid')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model


class KerasBiLSTMClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()

        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                # weights=[embedding_matrix] if embedding_matrix is not None else None,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True
            )
        )

        # Define parameter
        n_lstm = 128
        drop_lstm = 0.2
        model.add(layers.Bidirectional(layers.LSTM(n_lstm, return_sequences=False)))
        model.add(layers.Dropout(drop_lstm))

        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model


class KerasGRUClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()

        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                # weights=[embedding_matrix] if embedding_matrix is not None else None,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True
            )
        )

        model.add(layers.SpatialDropout1D(0.2))
        model.add(layers.GRU(128, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model


class KerasCNNandLSTMClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()

        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                # weights=[embedding_matrix] if embedding_matrix is not None else None,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True
            )
        )

        model.add(layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(100))
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model


class KerasHANClassifier(KerasClassifier):
    max_sent_num = 50

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        l2_reg = None

        embedding_layer = layers.Embedding(
            vocab_size, embedding_dim,
            # weights=[embedding_matrix] if embedding_matrix is not None else None,
            weights=[embedding_matrix],
            input_length=max_seq_len,
            trainable=False
        )
        # Words level attention model
        word_input = layers.Input(shape=(max_seq_len,), dtype='float32')
        word_sequences = embedding_layer(word_input)
        word_lstm = layers.Bidirectional(layers.LSTM(15, return_sequences=True, kernel_regularizer=l2_reg))(
            word_sequences)
        word_dense = layers.TimeDistributed(layers.Dense(20, kernel_regularizer=l2_reg))(word_lstm)
        word_att = AttentionWithContext()(word_dense)
        wordEncoder = tf.keras.Model(word_input, word_att)
        # Sentence level attention model
        sent_input = layers.Input(shape=(KerasHANClassifier.max_sent_num, max_seq_len), dtype='float32')
        sent_encoder = layers.TimeDistributed(wordEncoder)(sent_input)
        sent_lstm = layers.Bidirectional(layers.LSTM(15, return_sequences=True, kernel_regularizer=l2_reg))(
            sent_encoder)
        sent_dense = layers.TimeDistributed(layers.Dense(20, kernel_regularizer=l2_reg))(sent_lstm)
        sent_att = layers.Dropout(0.5)(AttentionWithContext()(sent_dense))
        preds = layers.Dense(1, activation='sigmoid')(sent_att)
        model = tf.keras.Model(sent_input, preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        if max_seq_len is None:
            raise Exception("max_seq_len can not be none in this model")

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        codes, labels = df['text'], df['label']

        codes_3d = np.zeros((len(codes), KerasHANClassifier.max_sent_num, max_seq_len), dtype='int32')
        for file_idx, file_code in enumerate(codes):
            for line_idx, line in enumerate(file_code.splitlines()):
                if line_idx >= KerasHANClassifier.max_sent_num:
                    continue

                X = embedding_model.text_to_indexes([line])[0]
                X = pad_sequences([X], padding='post', maxlen=max_seq_len)
                codes_3d[file_idx, line_idx, :] = X

        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        codes_3d, Y = sm.fit_resample(codes_3d, Y)

        model = cls.build_model(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            max_seq_len=max_seq_len
        )
        history = model.fit(
            codes_3d, Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        cls.plot_history(history)
        loss, accuracy = model.evaluate(codes_3d, Y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        return cls(model, embedding_model)

    def predict(self, df, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = df['text'], df['label']

        codes_3d = np.zeros((len(codes), KerasHANClassifier.max_sent_num, max_seq_len), dtype='int32')
        for file_idx, file_code in enumerate(codes):
            for line_idx, line in enumerate(file_code.splitlines()):
                if line_idx >= KerasHANClassifier.max_sent_num:
                    continue

                X = self.embedding_model.text_to_indexes([line])[0]
                X = pad_sequences([X], padding='post', maxlen=max_seq_len)
                codes_3d[file_idx, line_idx, :] = X

        return self.model.predict(codes_3d)
