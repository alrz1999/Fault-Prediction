from data.models import Project
from embedding.word2vec.word2vec import import_word2vec_model, export_word2vec_model


def import_model():
    project = Project("activemq")
    model = import_word2vec_model(project.name)
    print(model.wv.key_to_index["if"])


def export_model():
    project = Project("activemq")
    export_word2vec_model(project)


def main():
    import_model()


if __name__ == '__main__':
    main()
