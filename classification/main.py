from classification.cnn.cnn_baseline import KerasCNNClassifier
from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR
from data.models import Project
from classification.custom.custom_model import KerasClassifier, KerasCountVectorizerAndDenseLayer, \
    KerasTokenizerAndDenseLayer, SimpleKerasClassifierWithExternalEmbedding
from classification.mlp.mlp_baseline import MLPBaseLineClassifier
from classification.BoW.BoW_baseline import (BOWBaseLineClassifier)
from embedding.preprocessing.token_extraction import CustomTokenExtractor, ASTTokenizer, ASTExtractor

from embedding.word2vec.word2vec import GensimWord2VecModel, KerasTokenizer, SklearnCountTokenizer, KerasTextVectorizer
from pipeline.classification.classifier import ClassifierTrainingStage, PredictingClassifierStage
from pipeline.datas.file_level import LineLevelToFileLevelDatasetMapperStage, FileLevelDatasetImporterStage
from pipeline.datas.line_level import LineLevelDatasetImporterStage
from pipeline.embedding.embedding_model import EmbeddingModelImporterStage, EmbeddingModelTrainingStage, \
    IndexToVecMatrixAdderStage
from pipeline.evaluation.evaluation import EvaluationStage
from pipeline.models import Pipeline, StageData


def get_data_importer_pipeline_data(dataset_importer, metadata=None):
    training_data_importer_stages = [
        LineLevelDatasetImporterStage(dataset_importer),
        LineLevelToFileLevelDatasetMapperStage(),
    ]
    training_data_importer_pipeline_data = Pipeline(training_data_importer_stages).run(metadata)
    return training_data_importer_pipeline_data


def get_embedding_pipeline_data(embedding_cls, embedding_dim, project, token_extractor, training_data):
    embedding_stages = [
        EmbeddingModelTrainingStage(embedding_cls, project.name, embedding_dim, token_extractor,
                                    perform_export=False),
    ]
    embedding_pipeline_data = Pipeline(embedding_stages).run(training_data)
    return embedding_pipeline_data


def get_classifier_pipeline_data(classifier_cls, project, training_data):
    classifier_stages = [
        IndexToVecMatrixAdderStage(),
        ClassifierTrainingStage(
            classifier_cls,
            project.get_train_release().release_name,
            perform_export=False
        )
    ]
    training_pipeline_data = Pipeline(classifier_stages).run(training_data)
    return training_pipeline_data


def evaluate_classifier(project, pipeline_data):
    for eval_release in project.get_eval_releases():
        classifier_prediction_stages = [
            LineLevelDatasetImporterStage(eval_release),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                eval_release.release_name,
                output_columns=['Bug'],
                new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                             'test': eval_release.release_name},
                perform_export=False
            ),
            EvaluationStage()
        ]

        Pipeline(classifier_prediction_stages).run(pipeline_data)


def classify(project, classifier_cls, embedding_cls, token_extractor, embedding_dim, max_seq_len, batch_size, epochs):
    metadata = StageData({
        'dataset_name': project.get_train_release().release_name,
        'embedding_dim': embedding_dim,
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'epochs': epochs,
        'token_extractor': token_extractor
    })

    pipeline_data = get_data_importer_pipeline_data(
        dataset_importer=project.get_train_release(),
        metadata=metadata
    )

    if embedding_cls is not None:
        pipeline_data = get_embedding_pipeline_data(
            embedding_cls=embedding_cls,
            embedding_dim=embedding_dim,
            project=project,
            token_extractor=token_extractor,
            training_data=pipeline_data
        )

    pipeline_data = get_classifier_pipeline_data(
        classifier_cls=classifier_cls,
        project=project,
        training_data=pipeline_data,
    )

    evaluate_classifier(project, pipeline_data)


def mlp_classifier(project):
    max_seq_len = None
    classify(
        project=project,
        classifier_cls=MLPBaseLineClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=True, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
    )


def bow_classifier(project):
    classify(
        project=project,
        classifier_cls=BOWBaseLineClassifier,
        embedding_cls=None,
        token_extractor=None,
        embedding_dim=50,
        max_seq_len=None,
        batch_size=64,
        epochs=8,
    )


def keras_count_vectorizer_and_dense_layer(project):
    max_seq_len = 50

    classify(
        project=project,
        classifier_cls=KerasCountVectorizerAndDenseLayer,
        embedding_cls=SklearnCountTokenizer,
        token_extractor=CustomTokenExtractor(True, max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
    )


def keras_tokenizer_and_dense_layer(project):
    max_seq_len = 600

    classify(
        project=project,
        classifier_cls=KerasTokenizerAndDenseLayer,
        embedding_cls=KerasTokenizer,
        token_extractor=CustomTokenExtractor(True, max_seq_len),
        embedding_dim=250,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8
    )


def keras_classifier(project):
    max_seq_len = 300

    classify(
        project=project,
        classifier_cls=KerasClassifier,
        embedding_cls=KerasTextVectorizer,
        token_extractor=CustomTokenExtractor(True, max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8
    )


def keras_cnn_classifier(project):
    max_seq_len = 200
    token_extractor = CustomTokenExtractor(to_lowercase=True, max_seq_len=max_seq_len)
    # token_extractor = ASTTokenizer(cross_project=False)
    # token_extractor = ASTExtractor()

    classify(
        project=project,
        classifier_cls=KerasCNNClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=token_extractor,
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=4
    )


def simple_keras_classifier_with_external_embedding(project):
    max_seq_len = 400

    classify(
        project=project,
        classifier_cls=SimpleKerasClassifierWithExternalEmbedding,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=True, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def generate_line_level_dfs(project):
    project.get_train_release().export_line_level_dataset()
    for release in project.get_eval_releases():
        release.export_line_level_dataset()


if __name__ == '__main__':
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    # generate_line_level_dfs(project)

    # mlp_classifier(project)
    # bow_classifier(project)
    # keras_count_vectorizer_and_dense_layer(project)
    # keras_tokenizer_and_dense_layer(project)
    # keras_classifier(project)
    # keras_cnn_classifier(project)
    simple_keras_classifier_with_external_embedding(project)
