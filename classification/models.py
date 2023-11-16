class ClassifierModel:

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
