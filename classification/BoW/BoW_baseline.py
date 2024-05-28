import os
import pickle
import re

from sklearn.linear_model import LogisticRegression

from classification.models import ClassifierModel
from config import BOW_SAVE_MODEL_DIR, BOW_SAVE_PREDICTION_DIR


class BOWBaseLineClassifier(ClassifierModel):
    def __init__(self, model, vectorizer, dataset_name):
        self.model = model
        self.vectorizer = vectorizer
        self.dataset_name = dataset_name

    @classmethod
    def from_training(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        dataset_name = metadata.get('dataset_name')
        max_seq_len = metadata.get('max_seq_len')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        # available methods: smote, adasyn, rus, tomek, nearmiss, smotetomek
        imbalanced_learn_method = metadata.get('imbalanced_learn_method')

        X, Y = cls.prepare_X_and_Y(train_dataset, embedding_model, max_seq_len)
        X, Y, _ = cls.get_balanced_data_and_class_weight_dict(X, Y, class_weight_strategy, imbalanced_learn_method)

        clf = LogisticRegression(solver='liblinear')
        clf.fit(X, Y)

        return cls(clf, embedding_model, dataset_name)

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
        embedding_model = metadata.get('embedding_model')
        max_seq_len = metadata.get('max_seq_len')

        X, _ = self.prepare_X_and_Y(dataset, embedding_model, max_seq_len)
        return self.model.predict(X)

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        return os.path.join(BOW_SAVE_PREDICTION_DIR, dataset_name + 'new.csv')
