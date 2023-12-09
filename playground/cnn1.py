from collections import Counter

import numpy as np
from keras import Sequential, layers
from keras.src.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

from classification.evaluation.evaluation import evaluate
from classification.models import ClassifierModel
from classification.utils import LineLevelToFileLevelDatasetMapper
from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project
from embedding.preprocessing.token_extraction import CustomTokenExtractor, ASTTokenizer, ASTExtractor
from embedding.models import KerasTokenizer, GensimWord2VecModel


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    # token_extractor = CustomTokenExtractor(False, None)
    # token_extractor = ASTExtractor()
    token_extractor = ASTTokenizer(False)
    vocab_counter = Counter()
    to_lowercase = False
    embedding_dim = 100
    train_embedding_layer_in_classification_training = False

    line_level_dataset = project.get_train_release().get_processed_line_level_dataset()
    train_docs, train_labels = LineLevelToFileLevelDatasetMapper.prepare_data(line_level_dataset, to_lowercase)
    for doc in train_docs:
        doc_tokens = token_extractor.extract_tokens(doc)
        vocab_counter.update(doc_tokens)

    print(f'vocab_size={len(vocab_counter)}')
    print(vocab_counter.most_common(30))

    # min_occurrence = 2
    # tokens = [k for k, c in vocab_counter.items() if c >= min_occurrence]
    # print(len(tokens))

    vocabs = set(vocab_counter.keys())

    embedding_model = GensimWord2VecModel.train(
        train_docs,
        metadata={'to_lowercase': to_lowercase, 'token_extractor': token_extractor, 'embedding_dim': embedding_dim}
    )
    # fit the tokenizer on the documents
    train_encoded_docs = embedding_model.text_to_indexes(train_docs)
    print(
        f"len(vocabs - set(tokenizer.word_index.keys())):  {len(vocabs - set(embedding_model.get_word_to_index_dict().keys()))}")
    print(f'tokenizer size: {embedding_model.get_vocab_size()}')

    # pad sequences
    max_length = max([len(encoded_doc) for encoded_doc in train_encoded_docs])
    Xtrain = pad_sequences(train_encoded_docs, maxlen=max_length, padding='post')
    Ytrain = np.array([1 if label == True else 0 for label in train_labels])
    # SMOTE
    # Scale
    Xtest, Ytest = get_x_y(max_length, project.get_validation_release(), to_lowercase, embedding_model)

    vocab_size = embedding_model.get_vocab_size()
    embedding_vectors = embedding_model.get_embedding_matrix(embedding_model.get_word_to_index_dict(), vocab_size,
                                                             embedding_dim)

    model = Sequential()
    model.add(layers.Embedding(
        vocab_size, embedding_dim,
        weights=[embedding_vectors],
        input_length=max_length,
        trainable=train_embedding_layer_in_classification_training)
    )
    # model.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(layers.Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    # model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # fit network
    # history = model.fit(Xtrain, Ytrain, epochs=10, verbose=2, validation_data=(Xtest, Ytest))
    history = model.fit(Xtrain, Ytrain, epochs=10, verbose=2, validation_split=0.2)
    ClassifierModel.plot_history(history)

    loss, acc = model.evaluate(Xtest, Ytest, verbose=0)
    print('Test Accuracy: %f' % (acc * 100))

    for release in project.get_eval_releases()[1:]:
        Xeval, Y_eval = get_x_y(max_length, release, to_lowercase, embedding_model)
        predictions = model.predict(Xeval)
        print(f'predictions = {predictions}')
        # Y_pred = list(map(bool, list(predictions)))
        evaluate(Y_eval, predictions)


def get_x_y(max_length, release, to_lowercase, embedding_model):
    line_level_dataset = release.get_processed_line_level_dataset()
    val_docs, val_labels = LineLevelToFileLevelDatasetMapper.prepare_data(line_level_dataset, to_lowercase)
    validation_encoded_docs = embedding_model.text_to_indexes(val_docs)
    Xtest = pad_sequences(validation_encoded_docs, maxlen=max_length, padding='post')
    Ytest = np.array([1 if label == True else 0 for label in val_labels])
    return Xtest, Ytest


if __name__ == '__main__':
    main()
