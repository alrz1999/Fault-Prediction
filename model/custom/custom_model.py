import os

import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
from tensorflow import keras

from config import KERAS_SAVE_PREDICTION_DIR
from model.models import ClassifierModel
from model.utils import create_tensorflow_dataset


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
    def build_model(cls, max_features, embedding_dim):
        # A integer input for vocab indices.
        inputs = tf.keras.Input(shape=(None,), dtype="int64")

        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        x = layers.Embedding(max_features, embedding_dim)(inputs)
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
        return model

    @classmethod
    def vectorize_text(cls, vectorize_layer, text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    @classmethod
    def train(cls, df, dataset_name, training_metadata=None):
        max_features = training_metadata.get('max_features', 20000)
        embedding_dim = training_metadata.get('embedding_dim', 128)
        sequence_length = training_metadata.get('sequence_length', 500)
        batch_size = training_metadata.get('batch_size', 32)

        model = cls.build_model(max_features, embedding_dim)
        vectorize_layer = TextVectorization(
            max_tokens=max_features,
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

        # Do async prefetching / buffering of the data for best performance on GPU.
        train_ds = train_ds.cache().prefetch(buffer_size=10)
        val_ds = val_ds.cache().prefetch(buffer_size=10)

        epochs = 3

        # Fit the model using the train and test datasets.
        model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        return KerasClassifier(model, vectorize_layer)

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
        test_ds = test_ds.cache().prefetch(buffer_size=10)

        self.model.evaluate(test_ds)

    def get_end_to_end_model(self):
        model = self.model
        # A string input
        inputs = tf.keras.Input(shape=(1,), dtype="string")
        # Turn strings into vocab indices
        indices = self.vectorize_layer(inputs)
        # Turn vocab indices into predictions
        outputs = model(indices)

        # Our end to end model
        end_to_end_model = tf.keras.Model(inputs, outputs)
        end_to_end_model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return end_to_end_model

    def evaluate_end_to_end_model(self, test_df, batch_size=32):
        end_to_end_model = self.get_end_to_end_model()

        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        end_to_end_model.evaluate(test_ds)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_SAVE_PREDICTION_DIR, dataset_name + '.csv')
