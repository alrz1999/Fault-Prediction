import enum

import pandas as pd

from classification.evaluation.evaluation import evaluate
from classification.models import ClassificationDataset, DatasetType
from classification.torch_classifier.classifiers import TorchClassifier, TorchHANClassifier
from classification.utils import LineLevelToFileLevelDatasetMapper
from config import ORIGINAL_FILE_LEVEL_DATA_DIR, LINE_LEVEL_DATA_SAVE_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project, AggregatedDatasetImporter
from classification.keras_classifier.classifiers import KerasClassifier, KerasDenseClassifier, \
    KerasDenseClassifierWithEmbedding, KerasDenseClassifierWithExternalEmbedding, KerasCNNClassifierWithEmbedding, \
    KerasCNNClassifier, KerasLSTMClassifier, KerasBiLSTMClassifier, KerasGRUClassifier, KerasCNNandLSTMClassifier, \
    KerasHANClassifier, KerasMAMLClassifier1, SiameseClassifier, ReptileClassifier, TripletNetwork, PrototypicalNetwork
from classification.mlp.mlp_baseline import MLPBaseLineClassifier
from classification.BoW.BoW_baseline import (BOWBaseLineClassifier)
from embedding.preprocessing.token_extraction import CustomTokenExtractor, ASTTokenizer, ASTExtractor, \
    CommaSplitTokenExtractor

from embedding.models import GensimWord2VecModel, KerasTokenizer, SklearnCountTokenizer, KerasTextVectorizer, \
    EmbeddingModel


class ClassificationType(enum.Enum):
    FILE_LEVEL = 'FILE_LEVEL'
    CLASS_LEVEL = 'CLASS_LEVEL'
    FUNCTION_LEVEL = 'FUNCTION_LEVEL'
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
    elif classification_type == ClassificationType.FUNCTION_LEVEL:
        return dataset_importer.get_processed_method_level_dataset()
    else:
        raise Exception(f'training_type {classification_type} is not supported')


def get_embedding_matrix(embedding_model: EmbeddingModel):
    if embedding_model is None:
        return None
    return embedding_model.get_embedding_matrix(
        word_index=embedding_model.get_word_to_index_dict(),
        vocab_size=embedding_model.get_vocab_size(),
        embedding_dim=embedding_model.get_embedding_dim()
    )


def get_embedding_model(embedding_cls: EmbeddingModel, metadata, train_dataset):
    if embedding_cls is None:
        return None

    return embedding_cls.train(
        texts=train_dataset.get_texts(),
        metadata=metadata
    )


def classify(train_dataset_name, train_dataset_importer, eval_dataset_importers,
             classifier_cls, embedding_cls, token_extractor, embedding_dim, max_seq_len, batch_size, epochs,
             to_lowercase=False, vocab_size=None, validation_dataset_importer=None):
    train_dataset = ClassificationDataset(import_dataset(train_dataset_importer, to_lowercase), dataset_type)
    validation_dataset = ClassificationDataset(import_dataset(validation_dataset_importer, to_lowercase), dataset_type)

    metadata = {
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
        'dropout_ratio': 0.5,
        'class_weight_strategy': None,  # up_weight_majority, up_weight_minority
        'imbalanced_learn_method': None,  # smote, adasyn, rus, tomek, nearmiss, smotetomek,
        'load_best_model': True
    }

    embedding_model = get_embedding_model(embedding_cls, metadata, train_dataset)
    metadata['embedding_model'] = embedding_model

    embedding_matrix = get_embedding_matrix(embedding_model)
    metadata['embedding_matrix'] = embedding_matrix

    classifier_model = classifier_cls.from_training(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        metadata=metadata,
    )

    for eval_dataset_importer in eval_dataset_importers:
        evaluation_dataset = ClassificationDataset(import_dataset(eval_dataset_importer, to_lowercase), dataset_type)
        predicted_probabilities = classifier_model.predict(evaluation_dataset, metadata=metadata)
        true_labels = evaluation_dataset.get_labels()
        evaluate(true_labels, predicted_probabilities)


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
        # token_extractor=CustomTokenExtractor(to_lowercase, max_seq_len),
        token_extractor=ASTTokenizer(False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_dense_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = False
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasDenseClassifierWithEmbedding,
        embedding_cls=KerasTokenizer,
        # token_extractor=CustomTokenExtractor(to_lowercase, max_seq_len),
        # token_extractor=ASTExtractor(False),
        token_extractor=ASTTokenizer(False),
        embedding_dim=30,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_dense_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasDenseClassifierWithExternalEmbedding,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTExtractor(False),
        # token_extractor=ASTTokenizer(False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = False
    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=ASTTokenizer(False),
        # token_extractor=CustomTokenExtractor(to_lowercase, max_seq_len),
        token_extractor=ASTExtractor(cross_project=False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        vocab_size=10000,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_cnn_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = False
    # token_extractor = CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len)
    token_extractor = ASTTokenizer(cross_project=False)
    # token_extractor = ASTExtractor(cross_project=False)

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNClassifierWithEmbedding,
        embedding_cls=GensimWord2VecModel,
        token_extractor=token_extractor,
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=12,
        vocab_size=10000,
        to_lowercase=to_lowercase,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        token_extractor=ASTExtractor(cross_project=False),
        embedding_dim=64,
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
        token_extractor=ASTExtractor(cross_project=False),
        embedding_dim=64,
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
        token_extractor=ASTExtractor(cross_project=False),
        embedding_dim=64,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_cnn_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = None
    to_lowercase = True

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasCNNandLSTMClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        # token_extractor=ASTTokenizer(False),
        token_extractor=ASTExtractor(cross_project=False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=32,
        epochs=10,
        validation_dataset_importer=eval_dataset_importers[0]
    )


def keras_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers):
    max_seq_len = 500
    to_lowercase = False

    classify(
        train_dataset_name=train_dataset_name,
        train_dataset_importer=train_dataset_importer,
        eval_dataset_importers=eval_dataset_importers,
        classifier_cls=KerasHANClassifier,
        embedding_cls=GensimWord2VecModel,
        # token_extractor=CustomTokenExtractor(to_lowercase=to_lowercase, max_seq_len=max_seq_len),
        # token_extractor=ASTTokenizer(False),
        token_extractor=ASTTokenizer(cross_project=False),
        embedding_dim=50,
        max_seq_len=max_seq_len,
        batch_size=8,
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


def get_cross_release_dataset():
    project = Project(
        name="poi",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    return project.get_train_release().release_name, project.get_train_release(), project.get_eval_releases()


def get_cross_project_dataset():
    train_releases = []
    eval_releases = []
    for project_name in Project.releases_by_project_name.keys():
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
            method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
        )
        train_releases.append(project.get_train_release())
        eval_releases.extend(project.get_eval_releases())
    return 'cross-project', AggregatedDatasetImporter(train_releases), eval_releases


def get_cross_project_2_dataset():
    train_releases = []
    eval_releases = []

    train_project = Project(
        name='activemq',
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    train_releases.append(train_project.get_train_release())

    for project_name in Project.releases_by_project_name.keys():
        if project_name == train_project.name:
            continue
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
            method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
        )
        eval_releases.append(project.get_validation_release())
    return 'cross-project', AggregatedDatasetImporter(train_releases), eval_releases


classification_type = ClassificationType.FILE_LEVEL
dataset_type = DatasetType.FILE_LEVEL

if __name__ == '__main__':
    train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_release_dataset()
    # train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_project_dataset()
    # train_dataset_name, train_dataset_importer, eval_dataset_importers = get_cross_project_2_dataset()

    # mlp_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # bow_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_dense_classifier_with_external_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    keras_cnn_classifier_with_embedding(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_bilstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_gru_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_cnn_lstm_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # keras_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # torch_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
    # torch_han_classifier(train_dataset_name, train_dataset_importer, eval_dataset_importers)
