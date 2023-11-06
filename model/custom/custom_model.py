import tensorflow as tf
from keras import layers
from keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
from tensorflow import keras

from data.models import LineLevelDatasetGenerator
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
    def __init__(self, line_level_dataset_generator):
        self.max_features = 20000
        self.embedding_dim = 128
        self.sequence_length = 500
        self.line_level_dataset_generator: LineLevelDatasetGenerator = line_level_dataset_generator
        self.model = self.build_model()
        self.vectorize_layer = TextVectorization(
            max_tokens=self.max_features,
            output_mode="int",
            output_sequence_length=self.sequence_length,
        )

    def build_model(self):
        # A integer input for vocab indices.
        inputs = tf.keras.Input(shape=(None,), dtype="int64")

        # Next, we add a layer to map those vocab indices into a space of dimensionality
        # 'embedding_dim'.
        x = layers.Embedding(self.max_features, self.embedding_dim)(inputs)
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

    def vectorize_text(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label

    def train(self):
        df = self.line_level_dataset_generator.get_output_dataset()
        train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

        batch_size = 32
        train_dataset = create_tensorflow_dataset(train_data, batch_size=batch_size, shuffle=True)
        val_dataset = create_tensorflow_dataset(val_data, batch_size=batch_size)

        text_ds = train_dataset.map(lambda x, y: x)
        # Let's call `adapt`:
        self.vectorize_layer.adapt(text_ds)

        # Vectorize the data.
        train_ds = train_dataset.map(self.vectorize_text)
        val_ds = val_dataset.map(self.vectorize_text)

        # Do async prefetching / buffering of the data for best performance on GPU.
        train_ds = train_ds.cache().prefetch(buffer_size=10)
        val_ds = val_ds.cache().prefetch(buffer_size=10)

        epochs = 3

        # Fit the model using the train and test datasets.
        self.model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    def evaluate(self, test_df, batch_size=32):
        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        test_ds = test_ds.map(self.vectorize_text)
        test_ds = test_ds.cache().prefetch(buffer_size=10)

        self.model.evaluate(test_ds)
