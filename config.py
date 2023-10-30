# config.py

import os

# Define PROJECT_ROOT as the root directory of your project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define other project-related constants here, if needed
DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'original')
DATE_SAVE_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'preprocessed_data')
FILE_LEVEL_DATA_DIR = os.path.join(DATA_ROOT_DIR, 'File-level')
LINE_LEVEL_DATA_DIR = os.path.join(DATA_ROOT_DIR, 'Line-level')

# Define paths for saving models and predictions
BOW_SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'output', 'model', 'BoW')
BOW_SAVE_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'BoW')

# Define paths for saving models and predictions
MLP_SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'output', 'model', 'mlp')
MLP_SAVE_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'mlp')

# Define path for the Word2Vec model
WORD2VEC_DIR = os.path.join(PROJECT_ROOT, 'output', 'Word2Vec_model')
