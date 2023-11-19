import os

import numpy as np
from imblearn.over_sampling import SMOTE
from keras import layers, Sequential
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from classification.models import ClassifierModel
from classification.utils import create_tensorflow_dataset
from config import KERAS_CNN_SAVE_PREDICTION_DIR


class KerasCNNClassifier(ClassifierModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len):
        model = Sequential()

        # Next, we add a layer to map those vocab indices into a space of dimensionality 'embedding_dim'.
        model.add(
            layers.Embedding(
                vocab_size, embedding_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=True
            )
        )
        # model.add(layers.Flatten())
        # Modified CNN layers similar to the provided architecture.
        model.add(layers.Conv1D(100, 5, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D())
        model.add(layers.Dropout(0.2))

        # Add another Conv1D layer for complexity
        model.add(layers.Conv1D(100, 5, padding="same", activation="relu"))
        model.add(layers.MaxPooling1D())
        model.add(layers.Dropout(0.2))

        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dropout(0.2))
        # Vanilla hidden layer:
        model.add(layers.Dense(100, activation="relu"))
        model.add(layers.Dropout(0.2))

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
        epochs = training_metadata.get('epochs')
        num_words = training_metadata.get('num_words')

        codes, labels = df['SRC'], df['Bug']

        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(codes)

        embedding_matrix = np.array(embedding_model.get_embedding_matrix(tokenizer.word_index, embedding_dim))
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
            epochs=epochs,
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

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_CNN_SAVE_PREDICTION_DIR, dataset_name + '.csv')
