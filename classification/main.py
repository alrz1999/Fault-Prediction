import enum

from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR
from data.models import Project, AggregatedDatasetImporter
from classification.keras_classifiers.classifiers import KerasClassifier, KerasDenseClassifier, \
    KerasDenseClassifierWithEmbedding, KerasDenseClassifierWithExternalEmbedding, KerasCNNClassifierWithEmbedding, \
    KerasCNNClassifier, KerasLSTMClassifier, KerasBiLSTMClassifier, KerasGRUClassifier, KerasCNNandLSTMClassifier, \
    KerasHANClassifier
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
                output_columns=['label'],
                # new_columns={'project': eval_dataset_importer.project_name, 'train': train_dataset_name,
                #              'test': eval_dataset_importer.release_name},
                perform_export=False
            ),
            EvaluationStage()
        ]

        Pipeline(classifier_prediction_stages).run(pipeline_data)


def classify(train_dataset_name, train_dataset_importer, eval_dataset_importers,
             classifier_cls, embedding_cls, token_extractor, embedding_dim, max_seq_len, batch_size, epochs,
             to_lowercase=False, vocab_size=None):
    metadata = StageData({
        'training_type': training_type.value,
        'dataset_type': dataset_type.value,
        'dataset_name': train_dataset_name,
        'embedding_dim': embedding_dim,
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'epochs': epochs,
        'token_extractor': token_extractor,
        'vocab_size': vocab_size,
        'to_lowercase': to_lowercase,
        'perform_k_fold_cross_validation': False
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
            # dataset_name=project.name, #TODO
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
    to_lowercase = False
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=MLPBaseLineClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
        to_lowercase=to_lowercase
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


def keras_dense_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 50
    to_lowercase = False
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasDenseClassifier,
        embedding_cls=SklearnCountTokenizer,
        # embedding_cls=KerasTextVectorizer,
        token_extractor=CustomTokenExtractor(to_lowercase, max_seq_len),
        # token_extractor=ASTTokenizer(False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
        to_lowercase=to_lowercase
    )


def keras_dense_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 600
    to_lowercase = False
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasDenseClassifierWithEmbedding,
        embedding_cls=KerasTokenizer,
        token_extractor=CustomTokenExtractor(to_lowercase, max_seq_len),
        embedding_dim=250,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=8,
        to_lowercase=to_lowercase
    )


def keras_dense_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 400
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasDenseClassifierWithExternalEmbedding,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 50

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=ASTTokenizer(False),
        # token_extractor=ASTTokenizer(True),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=10,
        vocab_size=10000,
        to_lowercase=False
    )


def keras_cnn_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 150
    to_lowercase = False
    token_extractor = CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len)
    # token_extractor = ASTTokenizer(cross_project=False)
    # token_extractor = ASTExtractor()

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifierWithEmbedding,
        embedding_cls=GensimWord2VecModel,
        token_extractor=token_extractor,
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=64,
        epochs=4,
        to_lowercase=to_lowercase
    )


def keras_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 50
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def keras_bilstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasBiLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def keras_gru_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasGRUClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def keras_cnn_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNandLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=4
    )


def keras_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasHANClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
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


def get_cross_project_2_dataset():
    train_releases = []
    eval_releases = []

    train_project = Project(
        name='activemq',
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    train_releases.append(train_project.get_train_release())

    for project_name in Project.releases_by_project_name.keys():
        if project_name == train_project.name:
            continue
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
        )
        eval_releases.append(project.get_validation_release())
    return 'cross-project', AggregatedDatasetImporter(train_releases), eval_releases


training_type = TrainingType.FILE_LEVEL
dataset_type = DatasetType.LINE_LEVEL

if __name__ == '__main__':
    # generate_line_level_dfs()

    # train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_release_dataset()
    train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_project_dataset()

    # mlp_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # bow_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_bilstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_gru_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
