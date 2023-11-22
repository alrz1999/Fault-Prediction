import enum

from classification.cnn.cnn_baseline import KerasCNNClassifier
from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR
from data.models import Project, AggregatedDatasetImporter
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


class TrainingType(enum.Enum):
    FILE_LEVEL = 'FILE_LEVEL'
    CLASS_LEVEL = 'CLASS_LEVEL'
    FUNCTION_LEVEL = 'FUNCTION_LEVEL'
    LINE_LEVEL = 'LINE_LEVEL'


class DatasetType(enum.Enum):
    FILE_LEVEL = 'FILE_LEVEL'
    LINE_LEVEL = 'LINE_LEVEL'


def get_data_importer_pipeline_stages(dataset_importer):
    if training_type == TrainingType.FILE_LEVEL:
        if dataset_type == DatasetType.LINE_LEVEL:
            training_data_importer_stages = [
                LineLevelDatasetImporterStage(dataset_importer),
                LineLevelToFileLevelDatasetMapperStage(),
            ]
        elif dataset_type == DatasetType.FILE_LEVEL:
            training_data_importer_stages = [
                FileLevelDatasetImporterStage(dataset_importer),
            ]
        else:
            raise Exception(f'dataset_type {dataset_type} is not supported for training_type {training_type}')
    elif training_type == TrainingType.LINE_LEVEL:
        if dataset_type == DatasetType.LINE_LEVEL:
            training_data_importer_stages = [
                LineLevelDatasetImporterStage(dataset_importer),
            ]
        else:
            raise Exception(f'dataset_type {dataset_type} is not supported for training_type {training_type}')
    else:
        raise Exception(f'training_type {training_type} is not supported')
    return training_data_importer_stages


def get_data_importer_pipeline_data(dataset_importer, metadata=None):
    training_data_importer_stages = get_data_importer_pipeline_stages(dataset_importer)
    training_data_importer_pipeline_data = Pipeline(training_data_importer_stages).run(metadata)
    return training_data_importer_pipeline_data


def get_embedding_pipeline_data(embedding_cls, embedding_dim, dataset_name, token_extractor, training_data):
    embedding_stages = [
        EmbeddingModelTrainingStage(embedding_cls, dataset_name, embedding_dim, token_extractor, perform_export=False),
    ]
    embedding_pipeline_data = Pipeline(embedding_stages).run(training_data)
    return embedding_pipeline_data


def get_classifier_pipeline_data(classifier_cls, train_dataset_name, training_data):
    classifier_stages = [
        IndexToVecMatrixAdderStage(),
        ClassifierTrainingStage(
            classifier_cls,
            train_dataset_name,
            perform_export=False
        )
    ]
    training_pipeline_data = Pipeline(classifier_stages).run(training_data)
    return training_pipeline_data


def evaluate_classifier(eval_dataset_importers, train_dataset_name, pipeline_data):
    for eval_dataset_importer in eval_dataset_importers:
        classifier_prediction_stages = [
            *get_data_importer_pipeline_stages(eval_dataset_importer),
            PredictingClassifierStage(
                eval_dataset_importer.release_name,
                output_columns=['Bug'],
                new_columns={'project': eval_dataset_importer.project_name, 'train': train_dataset_name,
                             'test': eval_dataset_importer.release_name},
                perform_export=False
            ),
            EvaluationStage()
        ]

        Pipeline(classifier_prediction_stages).run(pipeline_data)


def classify(train_dataset_name, train_dataset_importer, eval_dataset_importers,
             classifier_cls, embedding_cls, token_extractor, embedding_dim, max_seq_len, batch_size, epochs,
             vocab_size=None):
    metadata = StageData({
        'dataset_name': train_dataset_name,
        'embedding_dim': embedding_dim,
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'epochs': epochs,
        'token_extractor': token_extractor,
        'vocab_size': vocab_size
    })

    pipeline_data = get_data_importer_pipeline_data(
        dataset_importer=train_dataset_importer,
        metadata=metadata
    )

    if embedding_cls is not None:
        pipeline_data = get_embedding_pipeline_data(
            embedding_cls=embedding_cls,
            embedding_dim=embedding_dim,
            dataset_name=train_dataset_name,
            # dataset_name=project.name,
            token_extractor=token_extractor,
            training_data=pipeline_data
        )

    pipeline_data = get_classifier_pipeline_data(
        classifier_cls=classifier_cls,
        train_dataset_name=train_dataset_name,
        training_data=pipeline_data,
    )

    evaluate_classifier(eval_dataset_importers, train_dataset_name, pipeline_data)


def mlp_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=MLPBaseLineClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=True, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
    )


def bow_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=BOWBaseLineClassifier,
        embedding_cls=None,
        token_extractor=None,
        embedding_dim=50,
        max_seq_len=None,
        batch_size=64,
        epochs=8,
    )


def keras_count_vectorizer_and_dense_layer(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 50

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCountVectorizerAndDenseLayer,
        embedding_cls=SklearnCountTokenizer,
        token_extractor=CustomTokenExtractor(True, max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
    )


def keras_tokenizer_and_dense_layer(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 600

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasTokenizerAndDenseLayer,
        embedding_cls=KerasTokenizer,
        token_extractor=CustomTokenExtractor(True, max_seq_len),
        embedding_dim=250,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8
    )


def keras_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 300

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=ASTTokenizer(True),
        # token_extractor=ASTTokenizer(True),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
        vocab_size=10000
    )


def keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 150
    # token_extractor = CustomTokenExtractor(to_lowercase=True, max_seq_len=max_seq_len)
    token_extractor = ASTTokenizer(cross_project=False)
    # token_extractor = ASTExtractor()

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=token_extractor,
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=4
    )


def simple_keras_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 400

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=SimpleKerasClassifierWithExternalEmbedding,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=True, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def generate_line_level_dfs():
    for project_name in Project.releases_by_project_name.keys():
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
        )
        project.get_train_release().export_line_level_dataset()
        for release in project.get_eval_releases():
            release.export_line_level_dataset()


def get_cross_release_dataset():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    return project.get_train_release().release_name, project.get_train_release(), project.get_eval_releases()


def get_cross_project_dataset():
    train_releases = []
    eval_releases = []
    for project_name in Project.releases_by_project_name.keys():
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
        )
        train_releases.append(project.get_train_release())
        eval_releases.extend(project.get_eval_releases())
    return 'cross-project', AggregatedDatasetImporter(train_releases), eval_releases


training_type = TrainingType.LINE_LEVEL
dataset_type = DatasetType.LINE_LEVEL

if __name__ == '__main__':
    # generate_line_level_dfs()

    train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_release_dataset()
    # train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_project_dataset()

    mlp_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # bow_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_count_vectorizer_and_dense_layer(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_tokenizer_and_dense_layer(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # simple_keras_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
