# feature_extraction.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # You can use other methods as well

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, input_data):
        pass


class TfidfFeatureExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.vectorizer = TfidfVectorizer()  # You can configure this based on your needs

    def extract_features(self, input_data):
        # Assuming input_data is a list of strings
        features = self.vectorizer.transform(input_data)
        return features


# You can create other feature extraction classes (e.g., Word Embeddings, custom methods) as needed

class CustomFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    def fit(self, X, y=None):
        # Fit the feature extractor on your data
        self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        # Transform the input data using the learned feature extraction
        return self.vectorizer.transform(X)

class NeuralFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def build_model(self, input_dim):
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))  # Input layer
        model.add(keras.layers.Dense(64, activation='relu'))  # Custom layers
        model.add(keras.layers.Dense(32, activation='relu'))
        return model

    def fit(self, X, y=None):
        input_dim = X.shape[1]  # Assuming X is a NumPy array
        self.model = self.build_model(input_dim)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, X, epochs=10, batch_size=32)  # Autoencoder - reconstruct input data
        return self

    def transform(self, X, y=None):
        # Use the encoder part of the model to extract features
        encoder = keras.Model(inputs=self.model.input, outputs=self.model.layers[1].output)
        features = encoder.predict(X)
        return features
