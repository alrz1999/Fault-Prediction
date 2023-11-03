from data.models import Project
from embedding.word2vec.word2vec import GensimWord2VecModel
from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )

    model = GensimWord2VecModel(
        line_level_dataset_generator=project,
        dataset_name="activemq",
        embedding_dimension=50
    ).import_model()

    print(model.wv.key_to_index["if"])


if __name__ == '__main__':
    main()
