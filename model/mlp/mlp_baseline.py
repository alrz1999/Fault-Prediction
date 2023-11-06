import argparse
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from config import MLP_SAVE_MODEL_DIR, MLP_SAVE_PREDICTION_DIR
from data.models import ProjectRelease
from embedding.word2vec.word2vec import GensimWord2VecModel
from model.models import ClassifierModel
from model.utils import prepare_data

warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train', action='store_true')
arg.add_argument('-predict', action='store_true')
args = arg.parse_args()


class MLPBaseLineClassifier(ClassifierModel):
    def __init__(self, train_data):
        self.train_data: ProjectRelease = train_data
        self.scaler = StandardScaler()

    def train(self):
        train_df = self.train_data.get_line_level_dataset(True, False, False)
        train_code, train_label = prepare_data(train_df, True)

        word2vec_model = GensimWord2VecModel(
            line_level_dataset_generator=self.train_data,
            dataset_name=self.train_data.project_name
        ).import_model()

        train_embeddings = []
        for code in train_code:
            code_embeddings = [
                word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word
                in
                code.split()]
            code_embeddings = np.mean(code_embeddings, axis=0)
            train_embeddings.append(code_embeddings)
        train_embeddings = np.array(train_embeddings)

        Y = np.array([1 if label == True else 0 for label in train_label])

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(train_embeddings, Y)

        # Create an MLP classifier
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)  # Modify parameters as needed
        X_res_scaled = self.scaler.fit_transform(X_res)
        clf.fit(X_res_scaled, y_res)
        print('finished training model for', self.train_data.release_name)

        return clf

    def import_model(self):
        return pickle.load(open(self.get_save_model_path(), 'rb'))

    def export_model(self):
        clf = self.train()
        pickle.dump(clf, open(self.get_save_model_path(), 'wb'))

    def get_save_model_path(self):
        if not os.path.exists(MLP_SAVE_MODEL_DIR):
            os.makedirs(MLP_SAVE_MODEL_DIR)
        if not os.path.exists(MLP_SAVE_PREDICTION_DIR):
            os.makedirs(MLP_SAVE_PREDICTION_DIR)

        return os.path.join(MLP_SAVE_MODEL_DIR, re.sub('-.*', '', self.train_data.release_name) + "-MLP-model.bin")

    def predict_defective_files_in_releases(self, eval_releases):
        clf = self.import_model()

        for rel in eval_releases:
            test_df = rel.get_line_level_dataset(True, False, False)

            test_code, train_label = prepare_data(test_df, True)  # Assuming prepare_data is correctly imported

            word2vec_model = GensimWord2VecModel(
                line_level_dataset_generator=self.train_data,
                dataset_name=self.train_data.project_name
            ).import_model()

            # Create Word2Vec embeddings for the code
            test_embeddings = []
            for code in test_code:
                code_embeddings = [
                    word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for
                    word
                    in
                    code.split()]
                code_embeddings = np.mean(code_embeddings, axis=0)
                test_embeddings.append(code_embeddings)
            test_embeddings = np.array(test_embeddings)

            # Standardize the input data
            X_scaled = self.scaler.transform(test_embeddings)

            Y_pred = list(map(bool, list(clf.predict(X_scaled))))

            result_df = pd.DataFrame()
            result_df['project'] = [rel.project_name] * len(Y_pred)
            result_df['train'] = [self.train_data.release_name] * len(Y_pred)
            result_df['test'] = [rel.release_name] * len(Y_pred)
            result_df['file-level-ground-truth'] = train_label
            result_df['prediction-label'] = Y_pred

            result_df.to_csv(self.get_result_dataset_path(rel.release_name), index=False)

            print('finish', rel.release_name)

    def get_result_dataset_path(self, dataset_name):
        return os.path.join(MLP_SAVE_PREDICTION_DIR, dataset_name + '.csv')
