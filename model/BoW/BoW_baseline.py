import re, os, pickle, warnings, sys, argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from config import DATE_SAVE_DIR, BOW_SAVE_MODEL_DIR, BOW_SAVE_PREDICTION_DIR
from model.utils import prepare_data

from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train', action='store_true')
arg.add_argument('-predict', action='store_true')

args = arg.parse_args()

if not os.path.exists(BOW_SAVE_MODEL_DIR):
    os.makedirs(BOW_SAVE_MODEL_DIR)

if not os.path.exists(BOW_SAVE_PREDICTION_DIR):
    os.makedirs(BOW_SAVE_PREDICTION_DIR)


# train_release is str
def train_model(train_release):
    train_df = train_release.import_line_level_df(DATE_SAVE_DIR)
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

    clf_path = os.path.join(BOW_SAVE_MODEL_DIR, re.sub('-.*', '', train_release.release) + "-BoW-model.bin")
    pickle.dump(clf, open(clf_path, 'wb'))
    vectorizer_path = os.path.join(BOW_SAVE_MODEL_DIR, re.sub('-.*', '', train_release.release) + "-vectorizer.bin")
    pickle.dump(vectorizer, open(vectorizer_path, 'wb'))

    print('finished training model for', train_release.project_name)

    count_vec_df = pd.DataFrame(X)
    count_vec_df.columns = vectorizer.get_feature_names_out()


# test_release is str
def predict_defective_files_in_releases(train_release, eval_releases):
    clf_path = os.path.join(BOW_SAVE_MODEL_DIR, re.sub('-.*', '', train_release.release) + "-BoW-model.bin")
    clf = pickle.load(open(clf_path, 'rb'))
    vectorizer_path = os.path.join(BOW_SAVE_MODEL_DIR, re.sub('-.*', '', train_release.release) + "-vectorizer.bin")
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    for rel in eval_releases:
        test_df = rel.import_line_level_df(DATE_SAVE_DIR)

        test_code, train_label = prepare_data(test_df, True)

        X = vectorizer.transform(test_code).toarray()

        Y_pred = list(map(bool, list(clf.predict(X))))
        Y_prob = clf.predict_proba(X)
        Y_prob = list(Y_prob[:, 1])

        result_df = pd.DataFrame()
        result_df['project'] = [rel.project_name] * len(Y_pred)
        result_df['train'] = [train_release.release] * len(Y_pred)
        result_df['test'] = [rel.release] * len(Y_pred)
        result_df['file-level-ground-truth'] = train_label
        result_df['prediction-prob'] = Y_prob
        result_df['prediction-label'] = Y_pred

        result_df.to_csv(os.path.join(BOW_SAVE_PREDICTION_DIR, rel.release + '.csv'), index=False)

        print('finish', rel.release)