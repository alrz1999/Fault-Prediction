class ClassifierModel:
    def train(self):
        raise NotImplementedError()

    def import_model(self):
        raise NotImplementedError()

    def export_model(self):
        raise NotImplementedError()

    def get_save_model_path(self):
        raise NotImplementedError()
