import os

from keras import layers, Sequential
from sklearn.model_selection import train_test_split

from classification.models import ClassifierModel
from classification.utils import create_tensorflow_dataset
from config import KERAS_CNN_SAVE_PREDICTION_DIR


class KerasCNNClassifier(ClassifierModel):
    def __init__(self, model):
        self.model = model

    @classmethod
    def build_model(cls, max_features, embedding_dim):
        # A integer input for vocab indices.
        # inputs = tf.keras.Input(shape=(None, embedding_dim))

        # Next, we add a layer to map those vocab indices into a space of dimensionality 'embedding_dim'.
        # x = layers.Embedding(max_features, embedding_dim)(inputs)
        # x = inputs
        # x = layers.Dropout(0.5)(x)
        model = Sequential()
        # Modified CNN layers similar to the provided architecture.
        model.add(layers.Conv1D(100, 5, padding="same", activation="relu", input_shape=(embedding_dim, 1)))
        model.add(layers.MaxPooling1D())
        model.add(layers.Dropout(0.5))

        # Add another Conv1D layer for complexity
        model.add(layers.Conv1D(100, 5, padding="same", activation="relu"))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dropout(0.5))

        # Vanilla hidden layer:
        model.add(layers.Dense(100, activation="relu", input_shape=(None, 500)))
        model.add(layers.Dropout(0.5))

        # Project onto a single unit output layer, and squash it with a sigmoid:
        model.add(layers.Dense(1, activation="sigmoid", name="predictions"))

        # Compile the model with binary crossentropy loss and an adam optimizer.
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    @classmethod
    def vectorize_text(cls, vectorize_layer, text, label):
        return vectorize_layer.get_embeddings(text), label

    @classmethod
    def train(cls, df, dataset_name, training_metadata=None):
        max_features = training_metadata.get('max_features', 20000)
        embedding_dim = training_metadata.get('embedding_dim', 128)
        batch_size = training_metadata.get('batch_size', 32)

        model = cls.build_model(max_features, embedding_dim)

        train_data, validation_data = train_test_split(df, test_size=0.2, random_state=42)

        train_ds = create_tensorflow_dataset(train_data, batch_size=batch_size, shuffle=True, key_column="embedding")
        val_ds = create_tensorflow_dataset(validation_data, batch_size=batch_size, key_column="embedding")

        epochs = 20

        model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        return KerasCNNClassifier(model)

    def predict(self, df, prediction_metadata=None):
        batch_size = prediction_metadata.get('batch_size', 32)

        test_ds = create_tensorflow_dataset(df, batch_size=batch_size, key_column="embedding")

        test_ds = test_ds.cache().prefetch(buffer_size=10)
        Y_pred = list(map(bool, list(self.model.predict(test_ds))))
        return Y_pred

    def evaluate(self, test_df, batch_size=32):
        test_ds = create_tensorflow_dataset(test_df, batch_size=batch_size)
        print(f"Number of batches in raw_test_ds: {test_ds.cardinality()}")

        self.model.evaluate(test_ds)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(KERAS_CNN_SAVE_PREDICTION_DIR, dataset_name + '.csv')
