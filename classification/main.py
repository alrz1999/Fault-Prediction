from classification.cnn.cnn_baseline import KerasCNNClassifier
from config import ORIGINAL_FILE_LEVEL_DATA_DIR, PREPROCESSED_DATA_SAVE_DIR
from data.models import Project
from classification.custom.custom_model import KerasClassifier, SimpleKerasClassifier, \
    SimpleKerasClassifierWithTokenizer, SimpleKerasClassifierWithExternalEmbedding
from classification.mlp.mlp_baseline import MLPBaseLineClassifier
from classification.BoW.BoW_baseline import (BOWBaseLineClassifier)

from embedding.word2vec.word2vec import GensimWord2VecModel, GensimWord2VecModelIndexer
from pipeline.classification.classifier import TrainingClassifierStage, PredictingClassifierStage
from pipeline.data.file_level import LineLevelToFileLevelDatasetMapperStage
from pipeline.data.line_level import LineLevelDatasetLoaderStage, LineLevelTokenizerStage
from pipeline.embedding.embedding_model import EmbeddingColumnAdderStage, TrainingEmbeddingModelStage
from pipeline.evaluation.evaluation import EvaluationStage
from pipeline.pipeline import Pipeline


def mlp_classifier(project):
    embedding_stages = [
        # LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        # LineLevelToFileLevelDatasetMapperStage(),
        TrainingEmbeddingModelStage(GensimWord2VecModel, project.name, 50, import_data=True)
    ]

    embedding_model = Pipeline(embedding_stages).run()

    training_classifier_stage = [
        # LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        # LineLevelToFileLevelDatasetMapperStage(),
        # EmbeddingColumnAdderStage(embedding_model),
        TrainingClassifierStage(MLPBaseLineClassifier, project.get_train_release().release_name, import_data=True)
    ]

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            EmbeddingColumnAdderStage(embedding_model),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                             'test': eval_release.release_name},
                export_data=True
            ),
            EvaluationStage()
        ]

        output = Pipeline(prediction_classifier_stages).run()


def bow_classifier(project):
    training_classifier_stage = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        TrainingClassifierStage(BOWBaseLineClassifier, project.get_train_release().release_name)
    ]

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
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

        output = Pipeline(prediction_classifier_stages).run()


def simple_keras_classifier(project):
    training_classifier_stage = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        TrainingClassifierStage(SimpleKerasClassifier, project.get_train_release().release_name)
    ]

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                # new_columns={'project': project.name, 'train': project.get_train_release().release_name,
                #              'test': eval_release.release_name},
            ),
            EvaluationStage()
        ]

        output = Pipeline(prediction_classifier_stages).run()


def simple_keras_classifier_with_tokenizer(project):
    max_seq_len = 500
    batch_size = 64
    epochs = 15
    num_words = 5000
    embedding_dim = 100

    training_classifier_stage = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        TrainingClassifierStage(
            SimpleKerasClassifierWithTokenizer,
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

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
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

        output = Pipeline(prediction_classifier_stages).run()


def keras_classifier(project):
    training_classifier_stage = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        TrainingClassifierStage(KerasClassifier, project.get_train_release().release_name, training_metadata={
            'max_features': 20000,
            'embedding_dim': 128,
            'sequence_length': 500,
            'batch_size': 32
        })
    ]

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                prediction_metadata={
                    'batch_size': 32
                },
                export_data=True
            ),
            EvaluationStage()
        ]

        output = Pipeline(prediction_classifier_stages).run()


def keras_cnn_classifier(project):
    embedding_dim = 50
    embedding_stages = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelTokenizerStage(),
        TrainingEmbeddingModelStage(GensimWord2VecModel, project.name, embedding_dim)
    ]

    embedding_model = Pipeline(embedding_stages).run()

    training_classifier_stage = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        TrainingClassifierStage(KerasCNNClassifier, project.get_train_release().release_name, training_metadata={
            'vocab_size': len(embedding_model.model.wv.key_to_index),
            'embedding_dim': embedding_dim,
            'batch_size': 32,
            'embedding_matrix': embedding_model.get_embedding_matrix()
        })
    ]

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
            LineLevelToFileLevelDatasetMapperStage(),
            PredictingClassifierStage(
                classifier,
                eval_release.release_name,
                output_columns=['Bug'],
                prediction_metadata={
                    'batch_size': 32,
                    'embedding_matrix': embedding_model.get_embedding_matrix()
                },
                export_data=True
            ),
            EvaluationStage()
        ]

        output = Pipeline(prediction_classifier_stages).run()


def simple_keras_classifier_with_external_embedding(project):
    embedding_dim = 50
    max_seq_len = 500
    embedding_stages = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelTokenizerStage(max_seq_len=max_seq_len),
        TrainingEmbeddingModelStage(GensimWord2VecModel, project.name, embedding_dim)
    ]

    embedding_model = Pipeline(embedding_stages).run()

    training_classifier_stage = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelToFileLevelDatasetMapperStage(),
        TrainingClassifierStage(
            SimpleKerasClassifierWithExternalEmbedding,
            project.get_train_release().release_name,
            training_metadata={
                'embedding_dim': embedding_dim,
                'batch_size': 32,
                'embedding_model': embedding_model,
                'max_seq_len': max_seq_len
            })
    ]

    classifier = Pipeline(training_classifier_stage).run()

    for eval_release in project.get_eval_releases():
        prediction_classifier_stages = [
            LineLevelDatasetLoaderStage(eval_release.get_line_level_dataset_path()),
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

        output = Pipeline(prediction_classifier_stages).run()


if __name__ == '__main__':
    project = Project(
        name="camel",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )

    # keras_cnn_classifier(project)
    # keras_classifier(project)
    # bow_classifier(project)
    # mlp_classifier(project)
    # simple_keras_classifier(project)
    simple_keras_classifier_with_tokenizer(project)
    # simple_keras_classifier_with_external_embedding(project)
