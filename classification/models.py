import enum

import matplotlib.pyplot as plt

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
    def train(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
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
