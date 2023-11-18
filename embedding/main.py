from data.models import Project
from embedding.models import EmbeddingModel
from embedding.word2vec.word2vec import GensimWord2VecModel
from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from pipeline.datas.line_level import LineLevelDatasetImporterStage, LineLevelTokenizerStage
from pipeline.embedding.embedding_model import EmbeddingModelTrainingStage
from pipeline.pipeline import Pipeline


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    train_release = project.get_train_release()

    stages = [
        LineLevelDatasetImporterStage(train_release.get_line_level_dataset_path()),
        LineLevelTokenizerStage(),
        EmbeddingModelTrainingStage(GensimWord2VecModel, project.name, 50, import_data=True)
    ]

    model: EmbeddingModel = Pipeline(stages).run()
    print(model.get_embeddings(['if']))


if __name__ == '__main__':
    main()
