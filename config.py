# config.py

import os

# Define PROJECT_ROOT as the root directory of your project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define other project-related constants here, if needed
ORIGINAL_DATA_ROOT_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'original')
ORIGINAL_FILE_LEVEL_DATA_DIR = os.path.join(ORIGINAL_DATA_ROOT_DIR, 'File-level')
ORIGINAL_BUGGY_LINES_DATA_DIR = os.path.join(ORIGINAL_DATA_ROOT_DIR, 'Line-level')

PREPROCESSED_DATA_SAVE_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'preprocessed_data')
METHOD_LEVEL_DATA_SAVE_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'preprocessed_data', 'Method-level')

# Define paths for saving models and predictions
BOW_SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'output', 'model', 'BoW')
BOW_SAVE_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'BoW')

# Define paths for saving models and predictions
MLP_SAVE_MODEL_DIR = os.path.join(PROJECT_ROOT, 'output', 'model', 'mlp')
MLP_SAVE_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'mlp')

# Define paths for saving models and predictions
KERAS_SAVE_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'keras')

# Define paths for saving models and predictions
KERAS_CNN_SAVE_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'keras_cnn')

# Define paths for saving models and predictions
SIMPLE_KERAS_PREDICTION_DIR = os.path.join(PROJECT_ROOT, 'output', 'prediction', 'keras_simple')

# Define path for the Word2Vec model
WORD2VEC_DIR = os.path.join(PROJECT_ROOT, 'output', 'Word2Vec_model')
