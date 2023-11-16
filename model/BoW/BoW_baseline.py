import os
import pickle
import re

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from config import BOW_SAVE_MODEL_DIR, BOW_SAVE_PREDICTION_DIR
from model.models import ClassifierModel


class BOWBaseLineClassifier(ClassifierModel):
    def __init__(self, model, vectorizer, dataset_name):
        self.model = model
        self.vectorizer = vectorizer
        self.dataset_name = dataset_name

    @classmethod
    def train(cls, df, dataset_name, training_metadata=None):
        codes, labels = df['SRC'], df['Bug']
        vectorizer = CountVectorizer()
        vectorizer.fit(codes)
        X = vectorizer.transform(codes).toarray()
        train_feature = pd.DataFrame(X)
        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(train_feature, Y)

        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_res, y_res)

        return BOWBaseLineClassifier(clf, vectorizer, dataset_name)

    @classmethod
    def get_vectorizer_model_path(cls, dataset_name):
        return os.path.join(BOW_SAVE_MODEL_DIR,
                            re.sub('-.*', '', dataset_name) + "-vectorizer.bin")

    @classmethod
    def import_model(cls, dataset_name):
        clf = pickle.load(open(cls.get_model_save_path(dataset_name), 'rb'))
        vectorizer = pickle.load(open(cls.get_vectorizer_model_path(dataset_name), 'rb'))
        return BOWBaseLineClassifier(clf, vectorizer, dataset_name)

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

    def predict(self, df, prediction_metadata=None):
        test_code, labels = df['SRC'], df['Bug']

        X = self.vectorizer.transform(test_code).toarray()

        Y_pred = list(map(bool, list(self.model.predict(X))))
        return Y_pred

        # Y_prob = self.model.predict_proba(X)
        # Y_prob = list(Y_prob[:, 1])
        #
        # result_df = pd.DataFrame()
        # result_df['project'] = [rel.project_name] * len(Y_pred)
        # result_df['train'] = [self.train_release_name] * len(Y_pred)
        # result_df['test'] = [rel.release_name] * len(Y_pred)
        # result_df['file-level-ground-truth'] = train_label
        # result_df['prediction-prob'] = Y_prob
        # result_df['prediction-label'] = Y_pred
        #
        # result_df.to_csv(self.get_result_dataset_path(rel.release_name), index=False)
        #
        # print('finish', rel.release_name)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(BOW_SAVE_PREDICTION_DIR, dataset_name + 'new.csv')
