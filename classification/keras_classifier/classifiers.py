import os
from pathlib import Path
from random import shuffle

import numpy as np
import tensorflow as tf
from keras import layers, Sequential
from keras.src import regularizers
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.optimizers import Adam
from keras.src.utils import pad_sequences, plot_model
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, cross_validate
from sklearn.utils import compute_class_weight
from tensorflow.keras import utils
from tensorflow.keras.models import load_model, save_model

from classification.keras_classifier.Reptile import ReptileDataset
from classification.keras_classifier.attention_with_context import AttentionWithContext
from classification.keras_classifier.mml import MAMLDataLoader, MAML, TaskGenerator
from classification.models import ClassifierModel, ClassificationDataset
from config import KERAS_SAVE_PREDICTION_DIR, SIMPLE_KERAS_PREDICTION_DIR, KERAS_CNN_SAVE_PREDICTION_DIR


class KerasClassifier(ClassifierModel):
    def __init__(self, classifier, embedding_model):
        self.classifier = classifier
        self.embedding_model = embedding_model

    def get_model_save_path(self, train_dataset_name):
        class_name = self.__class__.__name__
        return f'models/{class_name}-{train_dataset_name}.h5'

    def fit(self, X_train, Y_train, batch_size, class_weight_dict, train_dataset_name, epochs, X_valid, Y_valid):
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        model_checkpoint = ModelCheckpoint(filepath=self.get_model_save_path(train_dataset_name), monitor='val_loss',
                                           save_best_only=True)
        history = self.classifier.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            validation_data=(X_valid, Y_valid),
            callbacks=[early_stopping, model_checkpoint]
        )
        self.plot_history(history)

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')
        X_test, _ = self.prepare_X_and_Y(dataset, self.embedding_model, max_seq_len)
        return self.classifier.predict(X_test)

    def evaluate(self, X, Y, dataset_type='Training'):
        loss, accuracy = self.classifier.evaluate(X, Y, verbose=False)
        print(f"{dataset_type} Accuracy: " + "{:.4f}".format(accuracy))

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_training(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        train_dataset_name = metadata.get('dataset_name')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        vocab_size = embedding_model.get_vocab_size() if embedding_model else metadata.get('vocab_size')
        embedding_dim = embedding_model.get_embedding_dim() if embedding_model else metadata.get('embedding_dim')
        # available methods: smote, adasyn, rus, tomek, nearmiss, smotetomek
        imbalanced_learn_method = metadata.get('imbalanced_learn_method')
        show_classifier_model_summary = metadata.get('show_summary', True)
        load_best_model = metadata.get('load_best_model', False)

        X_train, Y_train = cls.prepare_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

        X_train, Y_train, class_weight_dict = cls.get_balanced_data_and_class_weight_dict(X_train, Y_train,
                                                                                          class_weight_strategy,
                                                                                          imbalanced_learn_method)

        X_valid, Y_valid = cls.prepare_X_and_Y(validation_dataset, embedding_model, max_seq_len)

        if metadata.get('perform_k_fold_cross_validation'):
            cls.k_fold_cross_validation(X_train, Y_train, batch_size, embedding_dim, embedding_matrix, epochs,
                                        max_seq_len, vocab_size)

        classifier = cls.build_classifier(
            vocab_size,
            embedding_dim,
            embedding_matrix,
            max_seq_len,
            X_train=X_train,
            Y_train=Y_train
        )

        if show_classifier_model_summary:
            classifier.summary()

        cls.export_model_plot(classifier)

        model = cls(classifier, embedding_model)
        model.fit(X_train, Y_train, batch_size, class_weight_dict, train_dataset_name, epochs, X_valid, Y_valid)

        if load_best_model:
            classifier = load_model(model.get_model_save_path(train_dataset_name))
            model = cls(classifier, embedding_model)

        model.evaluate(X_train, Y_train)
        return model

    @classmethod
    def k_fold_cross_validation(cls, X, Y, batch_size, embedding_dim, embedding_matrix, epochs, max_seq_len,
                                vocab_size):
        kf = KFold(n_splits=10)
        validation_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            model = cls.build_classifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                embedding_matrix=embedding_matrix,
                max_seq_len=max_seq_len,
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

    @classmethod
    def export_model_plot(cls, model):
        Path('./plots').mkdir(parents=True, exist_ok=True)
        plot_model(model, to_file=f'plots/{cls.__name__}.pdf', show_shapes=False, show_layer_names=True)


class KerasDenseClassifier(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model = Sequential()
        model.add(layers.Dense(512, input_dim=embedding_dim, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(SIMPLE_KERAS_PREDICTION_DIR, dataset_name + '.csv')


class KerasDenseClassifierWithEmbedding(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

        return model


class KerasDenseClassifierWithExternalEmbedding(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

        return model


class KerasCNNClassifier(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

    #     return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasCNNClassifierWithEmbedding(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

        return model

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_CNN_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasLSTMClassifier(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

        return model


class KerasBiLSTMClassifier(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

        n_lstm = 32
        drop_lstm = 0.2
        model.add(layers.Bidirectional(layers.LSTM(n_lstm, return_sequences=False)))
        model.add(layers.Dense(64, name='FC1'))
        model.add(layers.Dropout(drop_lstm))

        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

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
    #     return model

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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
    #     return model
    #
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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

        return model


class KerasHANClassifier(KerasClassifier):
    max_sent_num = 100

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
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
    def from_training(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        learning_rate = metadata.get('learning_rate')
        dropout_ratio = metadata.get('dropout_ratio')
        vocab_size = embedding_model.get_vocab_size() if embedding_model else metadata.get('vocab_size')
        embedding_dim = embedding_model.get_embedding_dim() if embedding_model else metadata.get('embedding_dim')

        if max_seq_len is None:
            raise Exception("max_seq_len can not be none in this model")

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

        model = cls.build_classifier(
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

        return self.classifier.predict(codes_3d)


class BalancedBaggingUnderSampleClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def from_training(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
                      metadata=None):
        embedding_model = metadata.get('embedding_model')
        max_seq_len = metadata.get('max_seq_len')

        X_train, Y_train = cls.prepare_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

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
    def from_training(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
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

        X_train, Y_train = cls.prepare_X_and_Y(train_dataset, embedding_model, max_seq_len)
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
                validation_data=cls.prepare_X_and_Y(validation_dataset, embedding_model, max_seq_len),
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
        #     validation_data=cls.prepare_X_and_Y(validation_dataset, embedding_model, max_seq_len),
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


class ReptileClassifier(KerasClassifier):

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        return SiameseClassifier.build_classifier(vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs)
        # return KerasCNNClassifierWithEmbedding.build_classifier(vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs)

    def fit(self, X_train, Y_train, batch_size, class_weight_dict, train_dataset_name, epochs, X_valid, Y_valid):
        learning_rate = 0.003
        meta_step_size = 0.25

        inner_batch_size = 25
        eval_batch_size = 25

        meta_iters = 300
        eval_iters = 5
        inner_iters = 4

        eval_interval = 1
        train_shots = 20
        shots = 5
        classes = 2

        model = self.classifier

        train_reptile_dataset = ReptileDataset(X_train, Y_train)
        validation_reptile_dataset = ReptileDataset(X_valid, Y_valid)

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

        save_model(self.classifier, self.get_model_save_path(train_dataset_name))

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        X_test, Y_test = self.prepare_X_and_Y(dataset, self.embedding_model, max_seq_len)

        test_reptile_dataset = ReptileDataset(X_test, Y_test)
        learning_rate = 0.003
        eval_batch_size = 25
        eval_iters = 5
        shots = 5
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        train_set, test_embeddings, test_labels = test_reptile_dataset.get_mini_dataset(
            eval_batch_size, eval_iters, shots, split=True
        )

        for images, labels in train_set:
            with tf.GradientTape() as tape:
                preds = self.classifier(images)
                loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(labels, axis=-1), preds)
            grads = tape.gradient(loss, self.classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.classifier.trainable_weights))
        test_preds = self.classifier.predict(test_embeddings)

        return self.classifier.predict(X_test)


class KerasMAMLClassifier1(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        return KerasCNNClassifierWithEmbedding.build_classifier(vocab_size, embedding_dim, embedding_matrix,
                                                                max_seq_len,
                                                                **kwargs)

    #
    # @classmethod
    # def maml_train_step(cls, model, task_batch, task_lr, meta_lr, num_adaptation_steps, build_model_wrapper):
    #     meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
    #
    #     # Meta-training step
    #     with tf.GradientTape(persistent=True) as meta_tape:
    #         meta_loss = 0
    #
    #         for task_data in task_batch:
    #             # Assume task_data is a tuple of (input, label)
    #             x, y = task_data
    #
    #             # Create a copy of the model for this task
    #             task_model = build_model_wrapper()
    #             task_model.set_weights(model.get_weights())
    #
    #             # Inner loop: Task-specific adaptation
    #             for _ in range(num_adaptation_steps):
    #                 with tf.GradientTape() as task_tape:
    #                     pred = task_model(tf.expand_dims(x, axis=0), training=True)
    #                     loss = tf.keras.losses.binary_crossentropy(np.array([[y]]), pred)
    #
    #                 gradients = task_tape.gradient(loss, task_model.trainable_variables)
    #                 if None in gradients:
    #                     raise ValueError("Some gradients are None. Check your model and loss function.")
    #                 task_optimizer = tf.keras.optimizers.Adam(learning_rate=task_lr)
    #                 task_optimizer.apply_gradients(zip(gradients, task_model.trainable_variables))
    #
    #             # Calculate the meta-loss
    #             pred = task_model(tf.expand_dims(x, axis=0), training=True)
    #             if meta_loss is None:
    #                 meta_loss = tf.keras.losses.binary_crossentropy(np.array([[y]]), pred)
    #             else:
    #                 meta_loss += tf.keras.losses.binary_crossentropy(np.array([[y]]), pred)
    #     if not model.trainable_variables:
    #         raise ValueError("No trainable variables found in the model.")
    #
    #     # Meta-update
    #     meta_gradients = meta_tape.gradient(meta_loss, model.trainable_variables)
    #     meta_optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))

    @classmethod
    def maml_train_step(cls, model, task_batch, task_lr, meta_lr, num_adaptation_steps):
        if not model.trainable_variables:
            raise ValueError("No trainable variables found in the model.")

        meta_optimizer = tf.keras.optimizers.Adam(learning_rate=meta_lr)
        task_optimizer = tf.keras.optimizers.Adam(learning_rate=task_lr)
        loss_object = tf.keras.losses.BinaryCrossentropy()

        # Initialize meta_loss as a 0-D Tensor
        meta_loss = tf.constant(0.0)

        # Meta-training step
        with tf.GradientTape(persistent=True) as meta_tape:
            import copy
            # Save original weights and optimizer state
            original_weights = copy.deepcopy(model.get_weights())

            for task_data in task_batch:
                x, y = task_data  # Assume task_data is a tuple of (input, label)

                # Inner loop: Task-specific adaptation
                for _ in range(num_adaptation_steps):
                    with tf.GradientTape() as task_tape:
                        pred = model(tf.expand_dims(x, axis=0), training=True)
                        loss = loss_object(np.array([[y]]), pred)

                    gradients = task_tape.gradient(loss, model.trainable_variables)
                    if None in gradients:
                        raise ValueError("Some gradients are None. Check your model and loss function.")

                    # Apply gradients using the task optimizer
                    task_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Calculate the meta-loss
                pred = model(tf.expand_dims(x, axis=0), training=True)
                meta_loss += loss_object(np.array([[y]]), pred)

                # Restore the original weights and optimizer state before the next task
                model.set_weights(original_weights)

        # Compute meta-gradients
        meta_gradients = meta_tape.gradient(meta_loss, model.trainable_variables)
        if any(g is None for g in meta_gradients):
            raise ValueError("Some meta-gradients are None. Check your computational graph and model.")

        # Apply meta-updates
        meta_optimizer.apply_gradients(zip(meta_gradients, model.trainable_variables))

        # Cleaning up the persistent tape
        del meta_tape

    def fit(self, X_train, Y_train, batch_size, class_weight_dict, train_dataset_name, epochs, X_valid, Y_valid):
        task_dataset = [(X_train[i], Y_train[i]) for i in range(len(X_train))]
        task_batches = TaskGenerator(task_dataset, 20, 5).generate()

        task_lr = 0.01
        meta_lr = 0.001
        num_adaptation_steps = 4

        for task_batch in task_batches:
            self.maml_train_step(self.classifier, task_batch, task_lr, meta_lr, num_adaptation_steps)


class KerasMAMLClassifier2(KerasClassifier):
    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        return KerasCNNClassifierWithEmbedding.build_classifier(vocab_size, embedding_dim, embedding_matrix,
                                                                max_seq_len,
                                                                **kwargs)

    def fit(self, X_train, Y_train, batch_size, class_weight_dict, train_dataset_name, epochs, X_valid, Y_valid):
        inner_lr = 0.01
        outer_lr = 0.001
        batch_size = 20
        val_batch_size = 20
        train_data = MAMLDataLoader(X_train, Y_train, batch_size, k_shot=10, q_query=4)
        val_data = MAMLDataLoader(X_valid, Y_valid, val_batch_size, k_shot=10, q_query=4)

        inner_optimizer = tf.keras.optimizers.SGD(inner_lr)
        outer_optimizer = tf.keras.optimizers.SGD(outer_lr)

        maml = MAML(self.classifier, 2)
        epochs = 3
        for e in range(epochs):

            train_progbar = utils.Progbar(train_data.steps)
            val_progbar = utils.Progbar(val_data.steps)
            print('\nEpoch {}/{}'.format(e + 1, epochs))

            train_meta_loss = []
            train_meta_acc = []
            val_meta_loss = []
            val_meta_acc = []

            for i in range(train_data.steps):
                batch_train_loss, acc = maml.train_on_batch(train_data.get_one_batch(),
                                                            inner_optimizer,
                                                            inner_step=1,
                                                            outer_optimizer=outer_optimizer)

                train_meta_loss.append(batch_train_loss)
                train_meta_acc.append(acc)
                train_progbar.update(i + 1, [('loss', np.mean(train_meta_loss)),
                                             ('accuracy', np.mean(train_meta_acc))])

            for i in range(val_data.steps):
                batch_val_loss, val_acc = maml.train_on_batch(val_data.get_one_batch(), inner_optimizer, inner_step=2)

                val_meta_loss.append(batch_val_loss)
                val_meta_acc.append(val_acc)
                val_progbar.update(i + 1, [('val_loss', np.mean(val_meta_loss)),
                                           ('val_accuracy', np.mean(val_meta_acc))])

            maml.meta_model.save_weights("maml.h5")


class SiameseClassifier(KerasClassifier):

    @classmethod
    def generate_balanced_pairs(cls, embeddings, labels):
        positive_pairs = []
        negative_pairs = []

        label_indices = {label: np.where(labels == label)[0] for label in set(labels)}

        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            # Select a positive example
            pos_indices = label_indices[label]
            pos_idx = np.random.choice([i for i in pos_indices if i != idx], 1)[0]
            positive_pairs.append([embedding, embeddings[pos_idx]])

            # Select a negative example
            neg_labels = [l for l in label_indices if l != label]
            neg_label = np.random.choice(neg_labels)
            neg_idx = np.random.choice(label_indices[neg_label], 1)[0]
            negative_pairs.append([embedding, embeddings[neg_idx]])

        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        all_labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        combined = list(zip(all_pairs, all_labels))
        shuffle(combined)
        all_pairs[:], all_labels[:] = zip(*combined)

        return np.array(all_pairs), np.array(all_labels)

    @classmethod
    def build_base_network(cls, vocab_size, embedding_dim, max_seq_len):
        input = layers.Input(shape=(max_seq_len,))
        x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len, mask_zero=True, trainable=True)(input)
        x = layers.Conv1D(300, 5, padding="same", activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        return tf.keras.Model(input, x)

    @classmethod
    def build_siamese_network(cls, base_network):
        input_shape = (None,)  # max_sequence_length is your preprocessed text length
        input_a = layers.Input(shape=input_shape)
        input_b = layers.Input(shape=input_shape)
        # Because we re-use the same instance `base_network`, the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        # Calculate the absolute difference between the embeddings
        distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = layers.Dense(1, activation='sigmoid')(distance)
        # Define the model that takes in two input texts and outputs the similarity score
        model = tf.keras.Model(inputs=[input_a, input_b], outputs=prediction)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        X_train = kwargs.get('X_train')
        Y_train = kwargs.get('Y_train')

        siamese_pairs, siamese_labels = cls.generate_balanced_pairs(X_train, Y_train)
        base_network = cls.build_base_network(vocab_size, embedding_dim, max_seq_len)
        siamese_network = cls.build_siamese_network(base_network)
        siamese_network.fit([siamese_pairs[:, 0], siamese_pairs[:, 1]], siamese_labels, batch_size=32, epochs=10)

        for layer in base_network.layers:
            layer.trainable = False

        # Define the classifier using the output of the base network
        real_input = layers.Input(shape=(None,))
        base_features = base_network(real_input)
        classification_output = layers.Dense(64, activation='relu')(base_features)
        classification_output = layers.Dense(1, activation='sigmoid')(classification_output)  # Binary classification
        # Combined model for classification
        classifier_model = tf.keras.Model(inputs=real_input, outputs=classification_output)
        classifier_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return classifier_model


class TripletNetwork(KerasClassifier):

    @classmethod
    def generate_balanced_triplets(cls, embeddings, labels):
        triplets = []

        label_indices = {label: np.where(labels == label)[0] for label in set(labels)}

        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            pos_indices = label_indices[label]
            pos_idx = np.random.choice([i for i in pos_indices if i != idx], 1)[0]

            neg_labels = [l for l in label_indices if l != label]
            neg_label = np.random.choice(neg_labels)
            neg_idx = np.random.choice(label_indices[neg_label], 1)[0]

            triplets.append([embedding, embeddings[pos_idx], embeddings[neg_idx]])

        shuffle(triplets)

        return np.array(triplets)

    @classmethod
    def build_base_network(cls, vocab_size, embedding_dim, max_seq_len):
        input = layers.Input(shape=(max_seq_len,))
        x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_len)(input)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        return tf.keras.Model(input, x)

    @classmethod
    def build_triplet_network(cls, base_network):
        input_shape = (None,)
        input_anchor = layers.Input(shape=input_shape)
        input_positive = layers.Input(shape=input_shape)
        input_negative = layers.Input(shape=input_shape)

        processed_anchor = base_network(input_anchor)
        processed_positive = base_network(input_positive)
        processed_negative = base_network(input_negative)

        model = tf.keras.Model(inputs=[input_anchor, input_positive, input_negative],
                               outputs=[processed_anchor, processed_positive, processed_negative])

        model.compile(loss=TripletNetwork.triplet_loss)

        return model

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha=0.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        positive_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        negative_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(positive_distance - negative_distance + alpha, 0.0)
        return loss

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        X_train = kwargs.get('X_train')
        Y_train = kwargs.get('Y_train')

        triplets = cls.generate_balanced_triplets(X_train, Y_train)
        base_network = cls.build_base_network(vocab_size, embedding_dim, max_seq_len)
        triplet_network = cls.build_triplet_network(base_network)

        dummy_labels = np.zeros((triplets.shape[0], 1))

        triplet_network.fit([triplets[:, 0], triplets[:, 1], triplets[:, 2]], dummy_labels, batch_size=32, epochs=10)

        for layer in base_network.layers:
            layer.trainable = False

        real_input = layers.Input(shape=(None,))
        base_features = base_network(real_input)
        classification_output = layers.Dense(64, activation='relu')(base_features)
        classification_output = layers.Dense(1, activation='sigmoid')(classification_output)

        classifier_model = tf.keras.Model(inputs=real_input, outputs=classification_output)
        classifier_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return classifier_model


class PrototypicalNetwork(KerasClassifier):

    @classmethod
    def generate_support_query_sets(cls, embeddings, labels, n_way, n_support, n_query):
        """
        Generate support and query sets for Prototypical Networks.
        """
        unique_labels = np.unique(labels)
        selected_labels = np.random.choice(unique_labels, size=n_way, replace=True)

        support_embeddings = []
        support_labels = []
        query_embeddings = []
        query_labels = []

        for label in selected_labels:
            label_indices = np.where(labels == label)[0]
            chosen_indices = np.random.choice(label_indices, size=n_support + n_query, replace=True)
            support_indices = chosen_indices[:n_support]
            query_indices = chosen_indices[n_support:]

            support_embeddings.append(embeddings[support_indices])
            query_embeddings.append(embeddings[query_indices])

            support_labels.extend([label] * n_support)
            query_labels.extend([label] * n_query)

        support_embeddings = np.vstack(support_embeddings)
        query_embeddings = np.vstack(query_embeddings)

        return support_embeddings, np.array(support_labels), query_embeddings, np.array(query_labels)

    @classmethod
    def build_base_network(cls, embedding_dim, max_seq_len):
        """
        Builds a simple network for feature extraction.
        """
        model = tf.keras.Sequential([
            layers.Input(shape=(max_seq_len,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(embedding_dim)
        ])
        return model

    @classmethod
    def compute_prototypes(cls, embeddings, labels, n_way):
        """
        Compute the prototype of each class in the support set.
        """
        prototypes = []
        for i in range(n_way):
            mask = tf.equal(labels, i)
            prototype = tf.reduce_mean(tf.boolean_mask(embeddings, mask), axis=0)
            prototypes.append(prototype)
        return tf.stack(prototypes)

    @classmethod
    def euclidean_distance(cls, a, b):
        """
        Compute the Euclidean distance between two tensors.
        """
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1))

    @classmethod
    def build_classifier(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        X_train = kwargs.get('X_train')
        Y_train = kwargs.get('Y_train')
        n_way, n_support, n_query = 2, 10, 5
        # Build the base network
        base_network = cls.build_base_network(embedding_dim, max_seq_len)

        # Training loop
        optimizer = tf.keras.optimizers.Adam()
        for epoch in range(10):  # Number of epochs
            support_embeddings, support_labels, query_embeddings, query_labels = cls.generate_support_query_sets(
                X_train, Y_train, n_way, n_support, n_query)

            with tf.GradientTape() as tape:
                # Embedding the support and query sets
                support_embeddings = base_network(support_embeddings)
                query_embeddings = base_network(query_embeddings)

                # Compute prototypes
                prototypes = cls.compute_prototypes(support_embeddings, support_labels, n_way)

                # Calculate distances and loss
                distances = tf.map_fn(lambda q_embedding: cls.euclidean_distance(q_embedding, prototypes),
                                      query_embeddings)
                log_p_y = tf.nn.log_softmax(-distances, axis=-1)
                correct_indices = tf.stack((tf.range(query_labels.shape[0]), query_labels), axis=1)
                loss = -tf.reduce_mean(tf.gather_nd(log_p_y, correct_indices))

            gradients = tape.gradient(loss, base_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, base_network.trainable_variables))

            if epoch % 1 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

        # Freeze the base network for feature extraction
        for layer in base_network.layers:
            layer.trainable = False

        # Define the classifier using the output of the base network
        real_input = layers.Input(shape=(None,))
        base_features = base_network(real_input)
        classification_output = layers.Dense(64, activation='relu')(base_features)
        classification_output = layers.Dense(1, activation='sigmoid')(classification_output)  # Binary classification
        classifier_model = tf.keras.Model(inputs=real_input, outputs=classification_output)
        classifier_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return classifier_model
