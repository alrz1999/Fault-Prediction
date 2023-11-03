import re, os, pickle, warnings, sys, argparse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# You may need to adjust the following imports according to your project's structure
from config import MLP_SAVE_MODEL_DIR, MLP_SAVE_PREDICTION_DIR, PREPROCESSED_DATA_SAVE_DIR
from embedding.word2vec.word2vec import GensimWord2VecModel
from model.utils import prepare_data

warnings.filterwarnings('ignore')

arg = argparse.ArgumentParser()
arg.add_argument('-dataset', type=str, default='activemq', help='software project name (lowercase)')
arg.add_argument('-train', action='store_true')
arg.add_argument('-predict', action='store_true')
args = arg.parse_args()

if not os.path.exists(MLP_SAVE_MODEL_DIR):
    os.makedirs(MLP_SAVE_MODEL_DIR)

if not os.path.exists(MLP_SAVE_PREDICTION_DIR):
    os.makedirs(MLP_SAVE_PREDICTION_DIR)

scaler = StandardScaler()  # Move this scaler definition to a higher scope


# train_release is str
def train_model(train_release):
    train_df = train_release.import_line_level_df(
        PREPROCESSED_DATA_SAVE_DIR)  # Assuming import_line_level_df is correctly imported
    train_code, train_label = prepare_data(train_df, True)  # Assuming prepare_data is correctly imported

    word2vec_model = GensimWord2VecModel(
        line_level_dataset_generator=train_release,
        dataset_name=train_release.project_name
    ).export_model()

    # Create Word2Vec embeddings for the code
    train_embeddings = []
    for code in train_code:
        code_embeddings = [
            word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word in
            code.split()]
        code_embeddings = np.mean(code_embeddings, axis=0)
        train_embeddings.append(code_embeddings)
    train_embeddings = np.array(train_embeddings)

    Y = np.array([1 if label == True else 0 for label in train_label])

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(train_embeddings, Y)

    # Create an MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)  # Modify parameters as needed
    X_res_scaled = scaler.fit_transform(X_res)
    clf.fit(X_res_scaled, y_res)

    clf_path = os.path.join(MLP_SAVE_MODEL_DIR, re.sub('-.*', '', train_release.release_name) + "-MLP-model.bin")
    pickle.dump(clf, open(clf_path, 'wb'))

    print('finished training model for', train_release)


# test_release is str
def predict_defective_files_in_releases(train_release, eval_releases):
    clf_path = os.path.join(MLP_SAVE_MODEL_DIR, re.sub('-.*', '', train_release.release_name) + "-MLP-model.bin")
    clf = pickle.load(open(clf_path, 'rb'))

    for rel in eval_releases:
        test_df = rel.import_line_level_df(
            PREPROCESSED_DATA_SAVE_DIR)  # Assuming import_line_level_df is correctly imported

        test_code, train_label = prepare_data(test_df, True)  # Assuming prepare_data is correctly imported

        word2vec_model = GensimWord2VecModel(
            line_level_dataset_generator=train_release,
            dataset_name=train_release.project_name
        ).import_model()

        # Create Word2Vec embeddings for the code
        test_embeddings = []
        for code in test_code:
            code_embeddings = [
                word2vec_model.wv[word] if word in word2vec_model.wv else np.zeros(word2vec_model.vector_size) for word
                in
                code.split()]
            code_embeddings = np.mean(code_embeddings, axis=0)
            test_embeddings.append(code_embeddings)
        test_embeddings = np.array(test_embeddings)

        # Standardize the input data
        X_scaled = scaler.transform(test_embeddings)

        Y_pred = list(map(bool, list(clf.predict(X_scaled))))

        result_df = pd.DataFrame()
        result_df['project'] = [rel.project_name] * len(Y_pred)
        result_df['train'] = [train_release] * len(Y_pred)
        result_df['test'] = [rel] * len(Y_pred)
        result_df['file-level-ground-truth'] = train_label
        result_df['prediction-label'] = Y_pred

        result_df.to_csv(os.path.join(MLP_SAVE_PREDICTION_DIR, rel.release_name + '.csv'), index=False)

        print('finish', rel)
