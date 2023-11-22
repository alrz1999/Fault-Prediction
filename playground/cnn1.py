from collections import Counter

import numpy as np
from keras import Sequential, layers
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

from classification.evaluation.evaluation import evaluate
from classification.models import ClassifierModel
from classification.utils import LineLevelToFileLevelDatasetMapper
from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from data.models import Project
from embedding.preprocessing.token_extraction import CustomTokenExtractor, ASTTokenizer, ASTExtractor


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    # token_extractor = CustomTokenExtractor(False, None)
    # token_extractor = ASTExtractor()
    token_extractor = ASTTokenizer(False)
    vocab_counter = Counter()
    to_lower_case = False

    line_level_dataset = project.get_train_release().get_processed_line_level_dataset()
    train_docs, train_labels = LineLevelToFileLevelDatasetMapper().prepare_data(line_level_dataset, to_lower_case)
    for doc in train_docs:
        doc_tokens = token_extractor.extract_tokens(doc)
        vocab_counter.update(doc_tokens)

    print(f'vocab_size={len(vocab_counter)}')
    print(vocab_counter.most_common(30))

    # min_occurrence = 2
    # tokens = [k for k, c in vocab_counter.items() if c >= min_occurrence]
    # print(len(tokens))

    vocabs = set(vocab_counter.keys())

    train_docs_tokens = []
    for doc in train_docs:
        doc_tokens = token_extractor.extract_tokens(doc)
        # doc_tokens = [token for token in doc_tokens if token in vocabs]
        train_docs_tokens.append(doc_tokens)

    tokenizer = Tokenizer(lower=to_lower_case)
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_docs_tokens)

    train_encoded_docs = tokenizer.texts_to_sequences(train_docs_tokens)
    print(f"len(vocabs - set(tokenizer.word_index.keys())):  {len(vocabs - set(tokenizer.word_index.keys()))}")
    print(f'tokenizer size: {len(tokenizer.word_index)}')

    # pad sequences
    max_length = max([len(encoded_doc) for encoded_doc in train_encoded_docs])
    Xtrain = pad_sequences(train_encoded_docs, maxlen=max_length, padding='post')
    Ytrain = np.array([1 if label == True else 0 for label in train_labels])
    # SMOTE
    # Scale
    Xtest, Ytest = get_x_y(max_length, project.get_validation_release(), to_lower_case, token_extractor, tokenizer)

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential()
    model.add(layers.Embedding(vocab_size, 100, input_length=max_length))
    model.add(layers.Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
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
        Xeval, Y_eval = get_x_y(max_length, release, to_lower_case, token_extractor, tokenizer)
        predictions = model.predict(Xeval)
        print(f'predictions = {predictions}')
        # Y_pred = list(map(bool, list(predictions)))
        Y_pred = list(list(predictions > 0.3))
        print(f'Y_pred = {Y_pred}')
        evaluate(Y_eval, Y_pred)

        # Calculate the ROC curve
        fpr, tpr, thresholds = roc_curve(Y_eval, predictions)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()


def get_x_y(max_length, release, to_lower_case, token_extractor, tokenizer):
    val_docs, val_labels = LineLevelToFileLevelDatasetMapper().prepare_data(
        release.get_processed_line_level_dataset(), to_lower_case)
    validation_docs_tokens = []
    for doc in val_docs:
        doc_tokens = token_extractor.extract_tokens(doc)
        # doc_tokens = [token for token in doc_tokens if token in vocabs]
        validation_docs_tokens.append(doc_tokens)
    validation_encoded_docs = tokenizer.texts_to_sequences(validation_docs_tokens)
    Xtest = pad_sequences(validation_encoded_docs, maxlen=max_length, padding='post')
    Ytest = np.array([1 if label == True else 0 for label in val_labels])
    return Xtest, Ytest


if __name__ == '__main__':
    main()
