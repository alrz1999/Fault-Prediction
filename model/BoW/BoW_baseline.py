import argparse
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from config import BOW_SAVE_MODEL_DIR, BOW_SAVE_PREDICTION_DIR
from data.models import ProjectRelease
from model.models import ClassifierModel
from model.utils import prepare_data

warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train', action='store_true')
arg.add_argument('-predict', action='store_true')

args = arg.parse_args()


class BOWBaseLineClassifier(ClassifierModel):
    def __init__(self, train_data):
        self.train_data: ProjectRelease = train_data

    def train(self):
        train_df = self.train_data.get_line_level_dataset(True, False, False)
        train_code, train_label = prepare_data(train_df, True)

        vectorizer = CountVectorizer()
        vectorizer.fit(train_code)
        X = vectorizer.transform(train_code).toarray()
        train_feature = pd.DataFrame(X)
        Y = np.array([1 if label == True else 0 for label in train_label])

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(train_feature, Y)

        clf = LogisticRegression(solver='liblinear')
        clf.fit(X_res, y_res)

        print('finished training model for', self.train_data.project_name)
        return clf, vectorizer
        # count_vec_df = pd.DataFrame(X)
        # count_vec_df.columns = vectorizer.get_feature_names_out()

    def get_vectorizer_model_path(self):
        return os.path.join(BOW_SAVE_MODEL_DIR,
                            re.sub('-.*', '', self.train_data.release_name) + "-vectorizer.bin")

    def import_model(self):
        clf, vectorizer = self.train()

        pickle.dump(clf, open(self.get_save_model_path(), 'wb'))
        pickle.dump(vectorizer, open(self.get_vectorizer_model_path(), 'wb'))

    def export_model(self):
        clf = pickle.load(open(self.get_save_model_path(), 'rb'))
        vectorizer = pickle.load(open(self.get_vectorizer_model_path(), 'rb'))
        return clf, vectorizer

    def get_save_model_path(self):
        if not os.path.exists(BOW_SAVE_MODEL_DIR):
            os.makedirs(BOW_SAVE_MODEL_DIR)

        if not os.path.exists(BOW_SAVE_PREDICTION_DIR):
            os.makedirs(BOW_SAVE_PREDICTION_DIR)

        return os.path.join(BOW_SAVE_MODEL_DIR, re.sub('-.*', '', self.train_data.release_name) + "-BoW-model.bin")

    def predict_defective_files_in_releases(self, eval_releases):
        clf, vectorizer = self.export_model()
        for rel in eval_releases:
            rel: ProjectRelease
            test_df = rel.get_line_level_dataset(True, False, False)

            test_code, train_label = prepare_data(test_df, True)

            X = vectorizer.transform(test_code).toarray()

            Y_pred = list(map(bool, list(clf.predict(X))))
            Y_prob = clf.predict_proba(X)
            Y_prob = list(Y_prob[:, 1])

            result_df = pd.DataFrame()
            result_df['project'] = [rel.project_name] * len(Y_pred)
            result_df['train'] = [self.train_data.release_name] * len(Y_pred)
            result_df['test'] = [rel.release_name] * len(Y_pred)
            result_df['file-level-ground-truth'] = train_label
            result_df['prediction-prob'] = Y_prob
            result_df['prediction-label'] = Y_pred

            result_df.to_csv(self.get_result_dataset_path(rel.release_name), index=False)

            print('finish', rel.release_name)

    def get_result_dataset_path(self, dataset_name):
        return os.path.join(BOW_SAVE_PREDICTION_DIR, dataset_name + '.csv')
