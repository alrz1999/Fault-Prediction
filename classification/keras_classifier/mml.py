import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses


class MAMLDataLoader:

    def __init__(self, X, Y, batch_size, k_shot=1, q_query=1):
        self.X = X
        self.Y = Y
        self.dataset_size = len(self.X)
        self.steps = self.dataset_size // batch_size

        self.k_shot = k_shot
        self.q_query = q_query
        self.meta_batch_size = batch_size

    def __len__(self):
        return self.steps

    def get_one_task_data(self):

        support_data = []
        query_data = []

        support_embeddings = []
        support_label = []
        query_embeddings = []
        query_label = []

        indices = np.random.choice(self.dataset_size, self.k_shot + self.q_query)

        for index in indices[:self.k_shot]:
            support_data.append((self.X[index], self.Y[index]))

        for index in indices[self.k_shot:]:
            query_data.append((self.X[index], self.Y[index]))

        # shuffle support set
        random.shuffle(support_data)
        for data in support_data:
            support_embeddings.append(data[0])
            support_label.append(data[1])

        # shuffle query set
        random.shuffle(query_data)
        for data in query_data:
            query_embeddings.append(data[0])
            query_label.append(data[1])

        return np.array(support_embeddings), np.array(support_label), np.array(query_embeddings), np.array(query_label)

    def get_one_batch(self):
        while True:
            batch_support_embedding = []
            batch_support_label = []
            batch_query_embedding = []
            batch_query_label = []

            for _ in range(self.meta_batch_size):
                support_embedding, support_label, query_embedding, query_label = self.get_one_task_data()
                batch_support_embedding.append(support_embedding)
                batch_support_label.append(support_label)
                batch_query_embedding.append(query_embedding)
                batch_query_label.append(query_label)

            yield np.array(batch_support_embedding), np.array(batch_support_label), \
                np.array(batch_query_embedding), np.array(batch_query_label)


class MAML:
    def __init__(self, model, num_classes):
        self.num_classes = num_classes
        self.meta_model = model

    def train_on_batch(self, train_data, inner_optimizer, inner_step, outer_optimizer=None):
        batch_acc = []
        batch_loss = []
        task_weights = []

        meta_weights = self.meta_model.get_weights()

        meta_support_embedding, meta_support_label, meta_query_embedding, meta_query_label = next(train_data)
        for support_embedding, support_label in zip(meta_support_embedding, meta_support_label):

            self.meta_model.set_weights(meta_weights)
            for _ in range(inner_step):
                with tf.GradientTape() as tape:
                    logits = self.meta_model(support_embedding, training=True)
                    loss = losses.binary_crossentropy(support_label.reshape(logits.shape), logits)
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, self.meta_model.trainable_variables)
                inner_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

            task_weights.append(self.meta_model.get_weights())

        with tf.GradientTape() as tape:
            for i, (query_embedding, query_label) in enumerate(zip(meta_query_embedding, meta_query_label)):
                self.meta_model.set_weights(task_weights[i])

                logits = self.meta_model(query_embedding, training=True)
                loss = losses.binary_crossentropy(query_label.reshape(logits.shape), logits)
                loss = tf.reduce_mean(loss)
                batch_loss.append(loss)

                acc = tf.cast(tf.argmax(logits, axis=-1) == query_label, tf.float32)
                acc = tf.reduce_mean(acc)
                batch_acc.append(acc)

            mean_acc = tf.reduce_mean(batch_acc)
            mean_loss = tf.reduce_mean(batch_loss)

        self.meta_model.set_weights(meta_weights)
        if outer_optimizer:
            grads = tape.gradient(mean_loss, self.meta_model.trainable_variables)
            outer_optimizer.apply_gradients(zip(grads, self.meta_model.trainable_variables))

        return mean_loss, mean_acc


class TaskGenerator:
    def __init__(self, data, n_tasks, batch_size):
        """
        A task generator for MAML training.

        :param data: list of tuples, each tuple is (embedding, label).
        :param n_tasks: int, the number of tasks to generate.
        :param batch_size: int, the size of batches in each task.
        """
        self.data = data
        self.n_tasks = n_tasks
        self.batch_size = batch_size

    def split_into_tasks(self):
        """Split the dataset into multiple tasks."""
        task_size = len(self.data) // self.n_tasks
        tasks = [self.data[i:i + task_size] for i in range(0, len(self.data), task_size) if
                 len(self.data[i:i + task_size]) > 0]
        return tasks

    def create_task_batches(self, tasks):
        """Create batches from tasks, each batch is a list of tuples (embedding, label)."""
        task_batches = []
        for task in tasks:
            # Shuffle the task data
            random.shuffle(task)
            n_batches = len(task) // self.batch_size
            for i in range(n_batches):
                batch = task[i * self.batch_size:(i + 1) * self.batch_size]
                task_batches.append(batch)
        return task_batches

    def generate(self):
        """Generate tasks and batches."""
        tasks = self.split_into_tasks()
        task_batches = self.create_task_batches(tasks)
        return task_batches
