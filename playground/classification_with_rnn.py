import numpy as np
import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# https://www.tensorflow.org/text/tutorials/text_classification_rnn

from classification.utils import LineLevelToFileLevelDatasetMapper
from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()


def get_tf_dataset(release, to_lowercase=False):
    line_level_dataset = release.get_processed_line_level_dataset()
    docs, labels = LineLevelToFileLevelDatasetMapper.prepare_data(line_level_dataset, to_lowercase)

    data = {
        'text': docs,
        'label': [1 if label == True else 0 for label in labels]
    }
    train_df = pd.DataFrame(data)
    return tf.data.Dataset.from_tensor_slices((train_df['text'].values, train_df['label'].values))


def one_lstm():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )

    train_dataset = get_tf_dataset(project.get_train_release())
    test_dataset = get_tf_dataset(project.get_validation_release())

    print(train_dataset.element_spec)
    for example, label in train_dataset.take(1):
        print('text: ', example.numpy())
        print('label: ', label.numpy())

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for example, label in train_dataset.take(1):
        print('texts: ', example.numpy()[:3])
        print()
        print('labels: ', label.numpy()[:3])

    VOCAB_SIZE = 5000
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    vocab = np.array(encoder.get_vocabulary())
    print(vocab[:20])

    encoded_example = encoder(example)[:3].numpy()
    print(encoded_example)

    for n in range(3):
        print("Original: ", example[n].numpy())
        print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
        print()
    #
    # model = tf.keras.Sequential([
    #     encoder,
    #     tf.keras.layers.Embedding(
    #         input_dim=len(encoder.get_vocabulary()),
    #         output_dim=16,
    #         # Use masking to handle the variable sequence lengths
    #         mask_zero=True),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
    #     tf.keras.layers.Dense(16, activation='relu'),
    #     tf.keras.layers.Dense(1, activation="sigmoid")
    # ])

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    print([layer.supports_masking for layer in model.layers])

    sample_text = ('The movie was cool. The animation and the graphics '
                   'were out of this world. I would recommend this movie.')
    predictions = model.predict(np.array([sample_text]))
    print(predictions[0])

    # predict on a sample text with padding

    padding = "the " * 2000
    predictions = model.predict(np.array([sample_text, padding]))
    print(predictions[0])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = model.fit(
        train_dataset, epochs=10,
        validation_data=test_dataset,
        # validation_steps=30
    )

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)

    sample_text = ('The movie was cool. The animation and the graphics '
                   'were out of this world. I would recommend this movie.')
    predictions = model.predict(np.array([sample_text]))


if __name__ == '__main__':
    one_lstm()
