import os
import pickle
import re

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import MLP_SAVE_MODEL_DIR, MLP_SAVE_PREDICTION_DIR
from classification.models import ClassifierModel


class MLPBaseLineClassifier(ClassifierModel):
    def __init__(self, model, scaler, dataset_name):
        self.model = model
        self.scaler = scaler
        self.dataset_name = dataset_name

    @classmethod
    def train(cls, df, dataset_name, metadata=None):
        embedding_model = metadata.get('embedding_model')

        codes, labels = df['SRC'], df['Bug']

        X = np.array(embedding_model.text_to_vec(codes))
        Y = np.array([1 if label == True else 0 for label in labels])

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, Y)

        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        scaler = StandardScaler()
        X_res_scaled = scaler.fit_transform(X_res)
        clf.fit(X_res_scaled, y_res)
        print('finished training model for', dataset_name)

        return cls(clf, scaler, dataset_name)

    @classmethod
    def import_model(cls, dataset_name):
        model = pickle.load(open(cls.get_model_save_path(dataset_name), 'rb'))
        scaler = pickle.load(open(cls.get_scalar_save_path(dataset_name), 'rb'))
        return cls(model, scaler, dataset_name)

    @classmethod
    def get_scalar_save_path(cls, dataset_name):
        return cls.get_model_save_path(dataset_name).replace('.bin', '_scaler.pkl')

    @classmethod
    def get_model_save_path(cls, dataset_name):
        if not os.path.exists(MLP_SAVE_MODEL_DIR):
            os.makedirs(MLP_SAVE_MODEL_DIR)
        if not os.path.exists(MLP_SAVE_PREDICTION_DIR):
            os.makedirs(MLP_SAVE_PREDICTION_DIR)

        return os.path.join(MLP_SAVE_MODEL_DIR, re.sub('-.*', '', dataset_name) + "-MLP-model.bin")

    def export_model(self):
        pickle.dump(self.model, open(self.get_model_save_path(self.dataset_name), 'wb'))
        pickle.dump(self.scaler, open(self.get_scalar_save_path(self.dataset_name), 'wb'))

    def predict(self, df, metadata=None):
        embedding_model = metadata.get('embedding_model')
        codes = df['SRC']

        embeddings = embedding_model.text_to_vec(codes)

        X_scaled = self.scaler.transform(embeddings)
        Y_pred = list(map(bool, list(self.model.predict(X_scaled))))

        return Y_pred

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(MLP_SAVE_PREDICTION_DIR, dataset_name + 'new.csv')
