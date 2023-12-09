from data.models import Project
from embedding.models import EmbeddingModel
from embedding.preprocessing.token_extraction import CustomTokenExtractor
from embedding.word2vec.word2vec import GensimWord2VecModel
from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from pipeline.datas.line_level import LineLevelDatasetImporterStage
from pipeline.embedding.embedding_model import EmbeddingModelTrainingStage
from pipeline.models import Pipeline, StageData


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    train_release = project.get_train_release()
    token_extractor = CustomTokenExtractor(to_lowercase=True, max_seq_len=None)
    embedding_dim = 50
    stages = [
        LineLevelDatasetImporterStage(train_release),
        EmbeddingModelTrainingStage(GensimWord2VecModel, project.name, embedding_dim, token_extractor,
                                    perform_export=False)
    ]

    pipeline_data = Pipeline(stages).run()
    model: EmbeddingModel = pipeline_data[StageData.Keys.EMBEDDING_MODEL.value]
    print(model.text_to_vec(['if']))


if __name__ == '__main__':
    main()
