import os

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras import layers, Sequential
from keras.layers import TextVectorization
from keras.src.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
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
    def __init__(self, model, vectorize_layer):
        self.model = model
        self.vectorize_layer = vectorize_layer

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
    def vectorize_text(cls, vectorize_layer, text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    @classmethod
    def train(cls, df, dataset_name, training_metadata=None):
        vocab_size = training_metadata.get('vocab_size', 20000)
        embedding_dim = training_metadata.get('embedding_dim', 128)
        sequence_length = training_metadata.get('sequence_length', 500)
        batch_size = training_metadata.get('batch_size', 32)

        model = cls.build_model(vocab_size, embedding_dim)
        vectorize_layer = TextVectorization(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=sequence_length,
        )
        train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

        train_dataset = create_tensorflow_dataset(train_data, batch_size=batch_size, shuffle=True)
        val_dataset = create_tensorflow_dataset(val_data, batch_size=batch_size)

        text_ds = train_dataset.map(lambda x, y: x)
        vectorize_layer.adapt(text_ds)

        train_ds = train_dataset.map(lambda x, y: cls.vectorize_text(vectorize_layer, x, y))
        val_ds = val_dataset.map(lambda x, y: cls.vectorize_text(vectorize_layer, x, y))

        epochs = 3

        # Fit the model using the train and test datasets.
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
        cls.plot_history(history)

        return cls(model, vectorize_layer)

    def predict(self, df, prediction_metadata=None):
        batch_size = prediction_metadata.get('batch_size', 32)

        test_ds = create_tensorflow_dataset(df, batch_size=batch_size)

        test_ds = test_ds.map(lambda x, y: self.vectorize_text(self.vectorize_layer, x, y))
        test_ds = test_ds.cache().prefetch(buffer_size=10)
        Y_pred = list(map(bool, list(self.model.predict(test_ds))))
        return Y_pred

    def evaluate(self, test_df, batch_size=32):
        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        test_ds = test_ds.map(lambda x, y: self.vectorize_text(self.vectorize_layer, x, y))

        self.model.evaluate(test_ds)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_SAVE_PREDICTION_DIR, dataset_name + '.csv')


class KerasCountVectorizerAndDenseLayer(ClassifierModel):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

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
    def train(cls, df, dataset_name, training_metadata=None):
        epochs = training_metadata.get('epochs', 4)
        batch_size = training_metadata.get('batch_size', 32)

        codes, labels = df['SRC'], df['Bug']

        vectorizer = CountVectorizer(max_df=0.7, min_df=0.002)
        vectorizer.fit(codes)
        print(len(vectorizer.vocabulary_))
        X = vectorizer.transform(codes).toarray()
        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(X, Y)

        model = cls.build_model(input_dim=X.shape[1])
        history = model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size
        )
        cls.plot_history(history)

        loss, accuracy = model.evaluate(X, Y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))

        return cls(model, vectorizer)

    def predict(self, df, prediction_metadata=None):
        codes, labels = df['SRC'], df['Bug']

        X = self.vectorizer.transform(codes).toarray()

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
    def train(cls, df, dataset_name, training_metadata=None):
        codes, labels = df['SRC'], df['Bug']
        max_seq_len = training_metadata.get('max_seq_len')
        batch_size = training_metadata.get('batch_size')
        epochs = training_metadata.get('epochs')
        num_words = training_metadata.get('num_words')
        embedding_dim = training_metadata.get('embedding_dim', 50)

        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(codes)

        X = tokenizer.texts_to_sequences(codes)
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
            batch_size=batch_size
        )
        cls.plot_history(history)

        loss, accuracy = model.evaluate(X, Y, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))

        return cls(model, tokenizer)

    def predict(self, df, prediction_metadata=None):
        test_code, labels = df['SRC'], df['Bug']
        max_seq_len = prediction_metadata.get('max_seq_len')

        X = self.vectorizer.texts_to_sequences(test_code)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)

        Y_pred = list(map(bool, list(self.model.predict(X))))
        return Y_pred


class SimpleKerasClassifierWithExternalEmbedding(ClassifierModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
    def train(cls, df, dataset_name, training_metadata=None):
        embedding_dim = training_metadata.get('embedding_dim')
        batch_size = training_metadata.get('batch_size')
        embedding_model = training_metadata.get('embedding_model')
        max_seq_len = training_metadata.get('max_seq_len')

        codes, labels = df['SRC'], df['Bug']

        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(codes)

        embedding_matrix = embedding_model.get_embedding_matrix(tokenizer.word_index, embedding_dim)
        X = tokenizer.texts_to_sequences(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)

        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(X, Y)

        model = cls.build_model(
            vocab_size=len(tokenizer.word_index) + 1,
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
        return cls(model, tokenizer)

    def predict(self, df, prediction_metadata=None):
        max_seq_len = prediction_metadata.get('max_seq_len')

        codes, labels = df['SRC'], df['Bug']

        X_test = self.tokenizer.texts_to_sequences(codes)
        X_test = pad_sequences(X_test, padding='post', maxlen=max_seq_len)

        Y_pred = list(map(bool, list(self.model.predict(X_test))))
        return Y_pred

    def evaluate(self, test_df, batch_size=32):
        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        self.model.evaluate(test_ds)
