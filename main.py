from embedding.preprocessing import token_extraction, text_preprocessing
from embedding.feature_extraction import feature_extraction
from model import model_creation, data_loading
from train_and_evaluate import train_model

# Step 1: Load the dataset
data = data_loading.load_data('data/dataset.csv')

# Step 2: Text Preprocessing and Token Extraction
preprocessed_data = text_preprocessing.preprocess(data)
tokens = token_extraction.extract_tokens(preprocessed_data)

# Step 3: Feature Extraction
features = feature_extraction.extract_features(tokens)

# Step 4: Feature Balancing
balanced_features, balanced_labels = balance_data.smote_oversample(features, labels)

# Step 5: Select a Model for Training
model = model_creation.create_custom_model()

# Step 6: Training and Evaluation
train_model(model, balanced_features, balanced_labels)
