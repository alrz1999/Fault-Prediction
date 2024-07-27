import numpy as np
import random
import tensorflow as tf


class ReptileDataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.embedding_shape = self.X.shape[1]
        self.data = {}
        for index in range(len(self.X)):
            embedding = self.X[index]
            label = self.Y[index]
            if label not in self.data:
                self.data[label] = []
            self.data[label].append(embedding)
        self.labels = list(self.data.keys())

    def get_mini_dataset(
            self, batch_size, repetitions, shots, split=False
    ):
        num_classes = 2
        temp_labels = np.zeros(shape=(num_classes * shots))
        embeddings = np.zeros(shape=(num_classes * shots, self.embedding_shape))
        if split:
            test_labels = np.zeros(shape=(num_classes,))
            test_embeddings = np.zeros(shape=(num_classes, self.embedding_shape))

        # Get a random subset of labels from the entire label set.
        label_subset = random.sample(self.labels, num_classes)
        for class_idx, class_obj in enumerate(label_subset):
            # Use enumerated index value as a temporary label for mini-batch in
            # few shot learning.
            temp_labels[class_idx * shots: (class_idx + 1) * shots] = class_idx
            # If creating a split dataset for testing, select an extra sample from each
            # label to create the test dataset.
            if split:
                test_labels[class_idx] = class_idx
                embeddings_to_split = random.choices(
                    self.data[label_subset[class_idx]], k=shots + 1
                )
                test_embeddings[class_idx] = embeddings_to_split[-1]
                embeddings[
                class_idx * shots: (class_idx + 1) * shots
                ] = embeddings_to_split[:-1]
            else:
                # For each index in the randomly selected label_subset, sample the
                # necessary number of images.
                embeddings[
                class_idx * shots: (class_idx + 1) * shots
                ] = random.choices(self.data[label_subset[class_idx]], k=shots)

        dataset = tf.data.Dataset.from_tensor_slices(
            (embeddings.astype(np.float32), temp_labels.astype(np.int32))
        )
        dataset = dataset.shuffle(100).batch(batch_size).repeat(repetitions)
        if split:
            return dataset, test_embeddings, test_labels
        return dataset
