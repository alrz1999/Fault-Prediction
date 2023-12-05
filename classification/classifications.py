import enum

import pandas as pd

from classification.torch_classifier.classifiers import TorchClassifier, TorchHANClassifier
from classification.utils import LineLevelToFileLevelDatasetMapper
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
from pipeline.embedding.embedding_model import EmbeddingModelImporterStage, EmbeddingModelTrainingStage, \
    IndexToVecMatrixAdderStage
from pipeline.evaluation.evaluation import EvaluationStage
from pipeline.models import Pipeline, StageData


class ClassificationType(enum.Enum):
    FILE_LEVEL = 'FILE_LEVEL'
    CLASS_LEVEL = 'CLASS_LEVEL'
    FUNCTION_LEVEL = 'FUNCTION_LEVEL'
    LINE_LEVEL = 'LINE_LEVEL'


class DatasetType(enum.Enum):
    FILE_LEVEL = 'FILE_LEVEL'
    LINE_LEVEL = 'LINE_LEVEL'


def import_dataset(dataset_importer, to_lowercase):
    if dataset_importer is None:
        return None

    replace_na_with_empty = True
    return_blank_lines = False
    return_test_file_lines = False
    return_comment_lines = False
    if classification_type == ClassificationType.FILE_LEVEL:
        if dataset_type == DatasetType.FILE_LEVEL:
            return dataset_importer.get_processed_file_level_dataset()
        elif dataset_type == DatasetType.LINE_LEVEL:
            line_level_df = dataset_importer.get_processed_line_level_dataset(
                replace_na_with_empty=replace_na_with_empty,
                return_blank_lines=return_blank_lines,
                return_test_file_lines=return_test_file_lines,
                return_comment_lines=return_comment_lines
            )
            text, label = LineLevelToFileLevelDatasetMapper.prepare_data(line_level_df, to_lowercase)
            data = {'text': text, 'label': label}
            return pd.DataFrame(data)
    elif classification_type == ClassificationType.LINE_LEVEL:
        return dataset_importer.get_processed_line_level_dataset(
            replace_na_with_empty=replace_na_with_empty,
            return_blank_lines=return_blank_lines,
            return_test_file_lines=return_test_file_lines,
            return_comment_lines=return_comment_lines
        )
    else:
        raise Exception(f'training_type {classification_type} is not supported')


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
        pipeline_data[StageData.Keys.EVALUATION_SOURCE_CODE_DF.value] = import_dataset(eval_dataset_importer, pipeline_data['to_lowercase'])
        classifier_prediction_stages = [
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
             to_lowercase=False, vocab_size=None, validation_dataset_importer=None):
    pipeline_data = StageData({
        'dataset_name': train_dataset_name,
        'embedding_dim': embedding_dim,
        'max_seq_len': max_seq_len,
        'batch_size': batch_size,
        'epochs': epochs,
        'token_extractor': token_extractor,
        'vocab_size': vocab_size,
        'to_lowercase': to_lowercase,
        'perform_k_fold_cross_validation': False,
        'learning_rate': 0.001,
        'dropout_ratio': 0.5
    })

    train_dataset = import_dataset(train_dataset_importer, to_lowercase)
    validation_dataset = import_dataset(validation_dataset_importer, to_lowercase)
    pipeline_data[StageData.Keys.TRAINING_SOURCE_CODE_DF.value] = train_dataset
    pipeline_data[StageData.Keys.VALIDATION_SOURCE_CODE_DF.value] = validation_dataset

    if classification_type == ClassificationType.FILE_LEVEL:
        pipeline_data[StageData.Keys.FILE_LEVEL_DF.value] = train_dataset
        pipeline_data[StageData.Keys.VALIDATION_FILE_LEVEL_DF.value] = validation_dataset
    elif classification_type == ClassificationType.LINE_LEVEL:
        pipeline_data[StageData.Keys.LINE_LEVEL_DF.value] = train_dataset
        pipeline_data[StageData.Keys.VALIDATION_LINE_LEVEL_DF.value] = validation_dataset

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
        batch_size=32,
        epochs=10,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
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
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
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
        batch_size=32,
        epochs=10,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
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
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
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
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = False
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=ASTTokenizer(False),
        # token_extractor=CustomTokenExtractor(to_lowercase, max_seq_len),
        token_extractor=ASTExtractor(cross_project=True),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        vocab_size=10000,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_cnn_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 300
    to_lowercase = False
    # token_extractor = CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len)
    # token_extractor = ASTTokenizer(cross_project=False)
    token_extractor = ASTExtractor(cross_project=True)

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifierWithEmbedding,
        embedding_cls=GensimWord2VecModel,
        token_extractor=token_extractor,
        embedding_dim=100,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=12,
        vocab_size=10000,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = True

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTExtractor(),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_bilstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasBiLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTExtractor(),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=5,
        validation_dataset_importer=eval_dataset_importers[0]
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
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTExtractor(),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_cnn_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 500
    to_lowercase = True

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNandLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTTokenizer(False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = True

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasHANClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTTokenizer(False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )

def torch_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = True

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=TorchClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTTokenizer(False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )

def torch_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 100
    to_lowercase = True

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=TorchHANClassifier,
        embedding_cls=GensimWord2VecModel,
        token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
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
        name="lucene-new",
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


classification_type = ClassificationType.FILE_LEVEL
dataset_type = DatasetType.FILE_LEVEL

if __name__ == '__main__':
    # generate_line_level_dfs()

    train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_release_dataset()
    # train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_project_dataset()
    # train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_project_2_dataset()

    # mlp_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # bow_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_bilstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_gru_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    torch_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # torch_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
