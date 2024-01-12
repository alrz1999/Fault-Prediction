import enum

import matplotlib.pyplot as plt
from keras.src.utils import pad_sequences
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from sklearn.utils import compute_class_weight

plt.style.use('ggplot')


class DatasetType(enum.Enum):
    FILE_LEVEL = 'FILE_LEVEL'
    LINE_LEVEL = 'LINE_LEVEL'
    FUNCTION_LEVEL = 'FUNCTION_LEVEL'


class ClassificationDataset:
    def __init__(self, text_label_df, dataset_type):
        self.text_label_df = text_label_df
        self.dataset_type = dataset_type
        self.embedding_matrix = None
        self.ast = None

    def get_texts(self):
        if self.dataset_type == DatasetType.FUNCTION_LEVEL:
            return self.text_label_df['ast']
        else:
            return self.text_label_df['text']

    def get_labels(self):
        return self.text_label_df['label']


class ClassifierModel:
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    @classmethod
    def plot_history(cls, history):
        acc = history.history['accuracy']
        if 'val_accuracy' in history.history:
            val_acc = history.history['val_accuracy']
        else:
            val_acc = None
        loss = history.history['loss']
        if 'val_loss' in history.history:
            val_loss = history.history['val_loss']
        else:
            val_loss = None
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        if val_acc:
            plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        if val_loss:
            plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    @classmethod
    def from_training(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
                      metadata=None):
        raise NotImplementedError()

    @classmethod
    def import_model(cls, dataset_name):
        raise NotImplementedError()

    @classmethod
    def get_model_save_path(cls, dataset_name):
        raise NotImplementedError()

    def export_model(self):
        raise NotImplementedError()

    def predict(self, dataset: ClassificationDataset, metadata=None):
        raise NotImplementedError()

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        raise NotImplementedError()

    @classmethod
    def prepare_X_and_Y(cls, classification_dataset, embedding_model, max_seq_len):
        codes, labels = classification_dataset.get_texts(), classification_dataset.get_labels()

        X = embedding_model.text_to_indexes(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        Y = np.array([1 if label == True else 0 for label in labels])
        return X, Y

    @classmethod
    def get_balanced_data_and_class_weight_dict(cls, X_train, Y_train, class_weight_strategy, imbalanced_learn_method):
        one_count = np.sum(Y_train == 1)
        zero_count = np.sum(Y_train == 0)
        if one_count > zero_count:
            minority_class_count = zero_count
            majority_class_count = one_count
            minor_class = 0
            major_class = 1
        else:
            minority_class_count = one_count
            majority_class_count = zero_count
            minor_class = 1
            major_class = 0
        print(f'{minor_class} minority_class_count: {minority_class_count}')
        print(f'{major_class} majority_class_count: {majority_class_count}')
        desired_majority_count = minority_class_count * 2
        sampling_strategy = {major_class: desired_majority_count, minor_class: minority_class_count}
        print(f'imbalanced_learn_method = {imbalanced_learn_method}')
        if imbalanced_learn_method == 'smote':
            sm = SMOTE(random_state=42)
            X_train, Y_train = sm.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'adasyn':
            adasyn = ADASYN(random_state=42)
            X_train, Y_train = adasyn.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'rus':
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, Y_train = rus.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'tomek':
            tomek = TomekLinks()
            X_train, Y_train = tomek.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'nearmiss':
            near_miss = NearMiss()
            X_train, Y_train = near_miss.fit_resample(X_train, Y_train)
        elif imbalanced_learn_method == 'smotetomek':
            smotetomek = SMOTETomek(random_state=42)
            X_train, Y_train = smotetomek.fit_resample(X_train, Y_train)
        if class_weight_strategy == 'up_weight_majority':
            if imbalanced_learn_method not in {'nearmiss', 'rus', 'tomek'}:
                raise Exception(f"imbalanced_learn_method {imbalanced_learn_method} is not a down-sampling so "
                                f"majority up-weighing is not allowed")
            class_weight_dict = {0: majority_class_count / desired_majority_count, 1: 1}
        elif class_weight_strategy == 'up_weight_minority':
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
            class_weight_dict = dict(enumerate(class_weights))
        else:
            class_weight_dict = None
        print(f'class_weight_dict = {class_weight_dict}')
        return X_train, Y_train, class_weight_dict
