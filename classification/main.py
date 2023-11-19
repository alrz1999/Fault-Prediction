from classification.cnn.cnn_baseline import KerasCNNClassifier
from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR
from data.models import Project
from classification.custom.custom_model import KerasClassifier, KerasCountVectorizerAndDenseLayer, \
    KerasTokenizerAndDenseLayer, SimpleKerasClassifierWithExternalEmbedding
from classification.mlp.mlp_baseline import MLPBaseLineClassifier
from classification.BoW.BoW_baseline import (BOWBaseLineClassifier)

from embedding.word2vec.word2vec import GensimWord2VecModel, GensimWord2VecModelIndexer
from pipeline.classification.classifier import ClassifierTrainingStage, PredictingClassifierStage
from pipeline.datas.file_level import LineLevelToFileLevelDatasetMapperStage
from pipeline.datas.line_level import LineLevelDatasetImporterStage, LineLevelTokenizerStage
from pipeline.embedding.embedding_model import EmbeddingModelImporterStage, EmbeddingModelTrainingStage, \
    EmbeddingAdderStage
from pipeline.evaluation.evaluation import EvaluationStage
from pipeline.models import Pipeline, StageData


def mlp_classifier(project):
    embedding_cls = GensimWord2VecModel
    embedding_dimension = 50
    classifier_cls = MLPBaseLineClassifier
    max_seq_len = None

    training_classifier_stage = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        LineLevelTokenizerStage(max_seq_len),
        EmbeddingModelTrainingStage(embedding_cls, project.name, embedding_dimension, perform_export=True),
        EmbeddingAdderStage(),
        ClassifierTrainingStage(classifier_cls, project.get_train_release().release_name, perform_export=False)
    ]

    training_pipeline_data = Pipeline(training_classifier_stage).run()
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            EmbeddingModelImporterStage(embedding_cls, project.name, embedding_dimension),
            EmbeddingAdderStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                             'test': eval_release.release_name},
                perform_export=False
            ),
            EvaluationStage()
        ]

        Pipeline(prediction_classifier_stages).run()


def bow_classifier(project):
    embedding_cls = None
    embedding_dimension = 50
    classifier_cls = BOWBaseLineClassifier
    max_seq_len = None

    training_classifier_stage = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        ClassifierTrainingStage(classifier_cls, project.get_train_release().release_name)
    ]

    training_pipeline_data = Pipeline(training_classifier_stage).run()
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                             'test': eval_release.release_name},
            ),
            EvaluationStage()
        ]

        Pipeline(prediction_classifier_stages).run()


def keras_count_vectorizer_and_dense_layer(project):
    embedding_cls = None
    embedding_dimension = 50
    classifier_cls = KerasCountVectorizerAndDenseLayer
    max_seq_len = None

    training_classifier_stage = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        ClassifierTrainingStage(
            classifier_cls,
            project.get_train_release().release_name,
            training_metadata={
                'epochs': 4,
                'batch_size': 64
            }
        )
    ]

    training_pipeline_data = Pipeline(training_classifier_stage).run()
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                             'test': eval_release.release_name},
            ),
            EvaluationStage()
        ]

        Pipeline(prediction_classifier_stages).run()


def keras_tokenizer_and_dense_layer(project):
    classifier_cls = KerasTokenizerAndDenseLayer
    max_seq_len = 600
    batch_size = 64
    epochs = 8
    num_words = 6000
    embedding_dim = 250

    training_classifier_stage = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        ClassifierTrainingStage(
            classifier_cls,
            project.get_train_release().release_name,
            training_metadata={
                'max_seq_len': max_seq_len,
                'batch_size': batch_size,
                'epochs': epochs,
                'num_words': num_words,
                'embedding_dim': embedding_dim
            }
        )
    ]

    training_pipeline_data = Pipeline(training_classifier_stage).run()
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                # new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                #              'test': eval_release.release_name},
                prediction_metadata={
                    'max_seq_len': max_seq_len
                }
            ),
            EvaluationStage()
        ]

        Pipeline(prediction_classifier_stages).run()


def keras_classifier(project):
    embedding_cls = None
    embedding_dimension = 50
    classifier_cls = KerasClassifier
    max_seq_len = 300

    training_classifier_stage = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        ClassifierTrainingStage(
            classifier_cls,
            project.get_train_release().release_name,
            training_metadata={
                'vocab_size': 5000,
                'embedding_dim': embedding_dimension,
                'sequence_length': max_seq_len,
                'batch_size': 32
            }
        )
    ]

    training_pipeline_data = Pipeline(training_classifier_stage).run()
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                prediction_metadata={
                    'batch_size': 32
                },
                perform_export=False
            ),
            EvaluationStage()
        ]

        Pipeline(prediction_classifier_stages).run()


def keras_cnn_classifier(project):
    embedding_cls = GensimWord2VecModel
    embedding_dim = 500
    classifier_cls = KerasCNNClassifier
    max_seq_len = 600
    epochs = 10

    embedding_training_stages = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        LineLevelTokenizerStage(),
        EmbeddingModelTrainingStage(embedding_cls, project.name, embedding_dim),
    ]

    embedding_pipeline_data = Pipeline(embedding_training_stages).run()
    embedding_model = embedding_pipeline_data[StageData.Keys.EMBEDDING_MODEL]

    classifier_training_stages = [
        ClassifierTrainingStage(
            classifier_cls,
            project.get_train_release().release_name,
            training_metadata={
                'embedding_dim': embedding_dim,
                'batch_size': 32,
                'embedding_model': embedding_model,
                'max_seq_len': max_seq_len,
                'epochs': epochs,
                'num_words': 5000,
            }
        )
    ]

    training_pipeline_data = Pipeline(classifier_training_stages).run(embedding_pipeline_data)
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        classifier_prediction_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                prediction_metadata={
                    'max_seq_len': max_seq_len,
                },
                perform_export=False
            ),
            EvaluationStage()
        ]

        output = Pipeline(classifier_prediction_stages).run()


def simple_keras_classifier_with_external_embedding(project):
    embedding_cls = GensimWord2VecModel
    embedding_dim = 50
    classifier_cls = SimpleKerasClassifierWithExternalEmbedding
    max_seq_len = 400

    embedding_training_stages = [
        # LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        # LineLevelTokenizerStage(max_seq_len=max_seq_len),
        # EmbeddingModelTrainingStage(embedding_cls, project.name, embedding_dim, perform_export=True),
        EmbeddingModelImporterStage(embedding_cls, project.name, embedding_dim)
    ]

    embedding_pipeline_data = Pipeline(embedding_training_stages).run()
    embedding_model = embedding_pipeline_data[StageData.Keys.EMBEDDING_MODEL]

    classifier_training_stages = [
        LineLevelDatasetImporterStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        ClassifierTrainingStage(
            classifier_cls,
            project.get_train_release().release_name,
            training_metadata={
                'embedding_dim': embedding_dim,
                'batch_size': 32,
                'embedding_model': embedding_model,
                'max_seq_len': max_seq_len
            })
    ]

    training_pipeline_data = Pipeline(classifier_training_stages).run(embedding_pipeline_data)
    classifier = training_pipeline_data[StageData.Keys.CLASSIFIER_MODEL]

    for eval_release in project.get_eval_releases():
        classifier_prediction_stages = [
            LineLevelDatasetImporterStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                prediction_metadata={
                    'batch_size': 32,
                    'max_seq_len': max_seq_len
                },
            ),
            EvaluationStage()
        ]

        Pipeline(classifier_prediction_stages).run()


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
