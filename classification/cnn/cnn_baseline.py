import os

import numpy as np
from imblearn.over_sampling import SMOTE
from keras import layers, Sequential
from keras.src.utils import pad_sequences

from classification.models import ClassifierModel
from classification.utils import create_tensorflow_dataset
from config import KERAS_CNN_SAVE_PREDICTION_DIR
from embedding.models import EmbeddingModel


class KerasCNNClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()

        # Next, we add a layer to map those vocab indices into a space of dimensionality 'embedding_dim'.
        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix] if embedding_matrix is not None else None,
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
    def train(cls, df, dataset_name, training_metadata=None):
        batch_size = training_metadata.get('batch_size')
        embedding_model: EmbeddingModel = training_metadata.get('embedding_model')
        max_seq_len = training_metadata.get('max_seq_len')
        epochs = training_metadata.get('epochs')
        embedding_matrix = training_metadata.get('embedding_matrix')

        codes, labels = df['SRC'], df['Bug']

        X = embedding_model.text_to_indexes(codes)
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dimension()

        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(X, Y)

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
            validation_split=0.2
        )
        cls.plot_history(history)

        return cls(model, embedding_model)

    def predict(self, df, prediction_metadata=None):
        max_seq_len = prediction_metadata.get('max_seq_len')

        codes, labels = df['SRC'], df['Bug']

        X_test = self.embedding_model.text_to_indexes(codes)

        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)

        Y_pred = list(map(bool, list(self.model.predict(X_test))))
        return Y_pred

    def evaluate(self, test_df, batch_size=32):
        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        self.model.evaluate(test_ds)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_CNN_SAVE_PREDICTION_DIR, dataset_name + '.csv')
