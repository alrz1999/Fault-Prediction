import os

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, OneSidedSelection, CondensedNearestNeighbour, NearMiss
from imblearn.combine import SMOTETomek
from keras import layers, Sequential
from keras.src import regularizers
from keras.src.optimizers import Adam
from keras.src.utils import pad_sequences
from sklearn.model_selection import KFold
from sklearn.utils import compute_class_weight

from classification.keras_classifiers.attention_with_context import AttentionWithContext
from classification.models import ClassifierModel
from config import KERAS_SAVE_PREDICTION_DIR, SIMPLE_KERAS_PREDICTION_DIR, KERAS_CNN_SAVE_PREDICTION_DIR


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

        sm = SMOTE(random_state=42)
        X_train, Y_train = sm.fit_resample(X_train, Y_train)

        # adasyn = ADASYN(random_state=42)
        # X_train, Y_train = adasyn.fit_resample(X_train, Y_train)

        # rus = RandomUnderSampler(random_state=42)
        # X_train, Y_train = rus.fit_resample(X_train, Y_train)

        # tomek = TomekLinks()
        # X_train, Y_train = tomek.fit_resample(X_train, Y_train)

        # smotetomek = SMOTETomek(random_state=42)
        # X_train, Y_train = smotetomek.fit_resample(X_train, Y_train)

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

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
        class_weight_dict = dict(enumerate(class_weights))
        print(f'class_weight_dict = {class_weight_dict}')

        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            validation_data=cls.get_X_and_Y(validation_dataset, embedding_model, max_seq_len)
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


class KerasDenseClassifier(KerasClassifier):
    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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
        return model


class KerasGRUClassifier(KerasClassifier):
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

        # model.add(layers.SpatialDropout1D(0.2))
        model.add(layers.GRU(64, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(64, name='FC1'))
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model


class KerasCNNandLSTMClassifier(KerasClassifier):
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

        model.add(layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Bidirectional(layers.GRU(16, return_sequences=False)))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
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
