import os

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import layers, Sequential
from keras.layers import TextVectorization
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import KERAS_SAVE_PREDICTION_DIR, SIMPLE_KERAS_PREDICTION_DIR
from classification.models import ClassifierModel
from classification.utils import create_tensorflow_dataset


class CustomModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(self.input_dim,)),  # Input layer matching feature vector shape
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.output_dim, activation='softmax'),  # Output layer
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions


class KerasClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim):
        # A integer input for vocab indices.
        inputs = tf.keras.Input(shape=(None,), dtype="int64")

        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        x = layers.Embedding(vocab_size, embedding_dim)(inputs)
        x = layers.Dropout(0.5)(x)

        # Conv1D + global max pooling
        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
        x = layers.GlobalMaxPooling1D()(x)

        # We add a vanilla hidden layer:
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

        model = tf.keras.Model(inputs, predictions)

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')

        codes, labels = df['SRC'], df['Bug']

        X = embedding_model.text_to_indexes(codes)
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dim()

        Y = np.array([1 if label == True else 0 for label in labels])

        model = cls.build_model(vocab_size, embedding_dim)
        history = model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        cls.plot_history(history)

        return cls(model, embedding_model)

    def predict(self, df, metadata=None):
        codes, labels = df['SRC'], df['Bug']

        X_test = self.embedding_model.text_to_indexes(codes)

        Y_pred = list(map(bool, list(self.model.predict(X_test))))

        return Y_pred

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasCountVectorizerAndDenseLayer(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, input_dim):
        model = Sequential()
        model.add(layers.Dense(512, input_dim=input_dim, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        epochs = metadata.get('epochs', 4)
        batch_size = metadata.get('batch_size', 32)
        embedding_model = metadata.get('embedding_model')

        codes, labels = df['SRC'], df['Bug']

        X = embedding_model.text_to_indexes(codes).toarray()
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dim()

        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(X, Y)

        model = cls.build_model(input_dim=X.shape[1])
        history = model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        cls.plot_history(history)

        loss, accuracy = model.evaluate(X, Y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))

        return cls(model, embedding_model)

    def predict(self, df, metadata=None):
        codes, labels = df['SRC'], df['Bug']

        X = self.embedding_model.text_to_indexes(codes).toarray()

        Y_pred = list(map(bool, list(self.model.predict(X))))
        return Y_pred

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(SIMPLE_KERAS_PREDICTION_DIR, dataset_name + '.csv')


class KerasTokenizerAndDenseLayer(KerasCountVectorizerAndDenseLayer):
    @classmethod
    def build_model(cls, input_shape):
        model = Sequential()
        model.add(layers.Dense(512, input_dim=input_shape, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    @classmethod
    def build_model_with_embedding(cls, input_dim, max_seq_len, embedding_dim=128):
        model = Sequential()
        model.add(
            layers.Embedding(
                input_dim=input_dim,
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

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        codes, labels = df['SRC'], df['Bug']
        max_seq_len = metadata.get('max_seq_len')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        embedding_model = metadata.get('embedding_model')

        X = embedding_model.text_to_indexes(codes)
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dim()

        X = pad_sequences(X, padding='post', maxlen=max_seq_len)

        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(X, Y)

        model = cls.build_model(input_shape=X.shape[1])
        # model = cls.build_model_with_embedding(
        #     input_dim=num_words + 1,
        #     max_seq_len=max_seq_len,
        #     embedding_dim=embedding_dim
        # )
        history = model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        cls.plot_history(history)

        loss, accuracy = model.evaluate(X, Y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))

        return cls(model, embedding_model)

    def predict(self, df, metadata=None):
        test_code, labels = df['SRC'], df['Bug']
        max_seq_len = metadata.get('max_seq_len')

        X = self.embedding_model.text_to_indexes(test_code)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)

        Y_pred = list(map(bool, list(self.model.predict(X))))
        return Y_pred


class SimpleKerasClassifierWithExternalEmbedding(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()
        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True)
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

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        batch_size = metadata.get('batch_size')
        embedding_model = metadata.get('embedding_model')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')

        codes, labels = df['SRC'], df['Bug']

        X = embedding_model.text_to_indexes(codes)
        vocab_size = embedding_model.get_vocab_size()
        embedding_dim = embedding_model.get_embedding_dim()

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
            epochs=20,
            batch_size=batch_size,
            validation_split=0.2
        )
        cls.plot_history(history)
        return cls(model, embedding_model)

    def predict(self, df, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        codes, labels = df['SRC'], df['Bug']

        X_test = self.embedding_model.text_to_indexes(codes)
        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)

        Y_pred = list(map(bool, list(self.model.predict(X_test))))
        return Y_pred

    def evaluate(self, test_df, batch_size=32):
        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        self.model.evaluate(test_ds)
