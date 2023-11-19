import matplotlib.pyplot as plt

plt.style.use('ggplot')


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
    def train(cls, df, dataset_name, training_metadata=None):
        raise NotImplementedError()

    @classmethod
    def import_model(cls, dataset_name):
        raise NotImplementedError()

    @classmethod
    def get_model_save_path(cls, dataset_name):
        raise NotImplementedError()

    def export_model(self):
        raise NotImplementedError()

    def predict(self, df, prediction_metadata=None):
        raise NotImplementedError()

    @classmethod
    def get_result_dataset_path(cls, dataset_name):
        raise NotImplementedError()
