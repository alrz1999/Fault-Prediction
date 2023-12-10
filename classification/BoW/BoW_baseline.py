import os
import pickle
import re

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from config import BOW_SAVE_MODEL_DIR, BOW_SAVE_PREDICTION_DIR
from classification.models import ClassifierModel


class BOWBaseLineClassifier(ClassifierModel):
    def __init__(self, model, vectorizer, dataset_name):
        self.model = model
        self.vectorizer = vectorizer
        self.dataset_name = dataset_name

    @classmethod
    def train(cls, train_dataset, validation_dataset=None, metadata=None):
        dataset_name = metadata.get('dataset_name')
        codes, labels = train_dataset.get_texts(), train_dataset.get_labels()
        vectorizer = CountVectorizer()
        vectorizer.fit(codes)
        X = vectorizer.transform(codes).toarray()
        train_feature = pd.DataFrame(X)
        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X, Y = sm.fit_resample(train_feature, Y)

        clf = LogisticRegression(solver='liblinear')
        clf.fit(X, Y)

        return cls(clf, vectorizer, dataset_name)

    @classmethod
    def get_vectorizer_model_path(cls, dataset_name):
        return os.path.join(BOW_SAVE_MODEL_DIR,
                            re.sub('-.*', '', dataset_name) + "-vectorizer.bin")

    @classmethod
    def import_model(cls, dataset_name):
        clf = pickle.load(open(cls.get_model_save_path(dataset_name), 'rb'))
        vectorizer = pickle.load(open(cls.get_vectorizer_model_path(dataset_name), 'rb'))
        return cls(clf, vectorizer, dataset_name)

    def export_model(self):
        pickle.dump(self.model, open(self.get_model_save_path(self.dataset_name), 'wb'))
        pickle.dump(self.vectorizer, open(self.get_vectorizer_model_path(self.dataset_name), 'wb'))

    @classmethod
    def get_model_save_path(cls, dataset_name):
        if not os.path.exists(BOW_SAVE_MODEL_DIR):
            os.makedirs(BOW_SAVE_MODEL_DIR)

        if not os.path.exists(BOW_SAVE_PREDICTION_DIR):
            os.makedirs(BOW_SAVE_PREDICTION_DIR)

        return os.path.join(BOW_SAVE_MODEL_DIR, re.sub('-.*', '', dataset_name) + "-BoW-model.bin")

    def predict(self, dataset, metadata=None):
        test_code, labels = dataset.get_texts(), dataset.get_labels()

        X = self.vectorizer.transform(test_code).toarray()

        return self.model.predict(X)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(BOW_SAVE_PREDICTION_DIR, dataset_name + 'new.csv')
