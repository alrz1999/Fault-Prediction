import random

import learn2learn as l2l
import numpy as np
import tensorflow as tf
import torch
from keras import layers
from keras.src.utils import pad_sequences
from torch import nn, optim
from torch.utils.data import DataLoader

from classification.keras_classifier.l2l import BinaryClassifier, FewShotTextDataset
from classification.models import ClassifierModel, ClassificationDataset


class L2LClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    @classmethod
    def build_model(cls, vocab_size, embedding_dim, embedding_matrix, max_seq_len, **kwargs):
        model_input = kwargs.get('model_input')

        embedding_layer = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_len,
            trainable=True,
            mask_zero=True
        )(model_input)

        conv_layer = layers.Conv1D(100, 5, padding="same", activation="relu")(embedding_layer)
        global_max_pooling = layers.GlobalMaxPooling1D()(conv_layer)
        dropout_layer = layers.Dropout(0.25)(global_max_pooling)
        dense_layer = layers.Dense(100, activation="relu")(dropout_layer)
        predictions = layers.Dense(1, activation="sigmoid")(dense_layer)

        model = tf.keras.Model(inputs=model_input, outputs=predictions)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.summary()
        return model

    @classmethod
    def from_training(cls, train_dataset: ClassificationDataset, validation_dataset: ClassificationDataset = None,
                      metadata=None):
        np.random.seed(42)
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        dataset_name = metadata.get('dataset_name')
        class_weight_strategy = metadata.get('class_weight_strategy')  # up_weight_majority, up_weight_minority
        imbalanced_learn_method = metadata.get(
            'imbalanced_learn_method')  # smote, adasyn, rus, tomek, nearmiss, smotetomek

        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        X_train, Y_train = cls.get_X_and_Y(train_dataset, embedding_model, max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]
            metadata['max_seq_len'] = max_seq_len

        model = BinaryClassifier(vocab_size=vocab_size, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix)
        ways = 2
        shots = 1
        meta_lr = 0.003
        fast_lr = 0.5
        meta_batch_size = 32
        adaptation_steps = 1
        num_iterations = 100
        cuda = True
        seed = 42
        meta_train_dataset = l2l.data.MetaDataset(FewShotTextDataset(X_train, Y_train))

        task_transforms = [
            l2l.data.transforms.NWays(meta_train_dataset, n=2),  # Binary Classification
            l2l.data.transforms.KShots(meta_train_dataset, k=2 * shots),  # Number of examples
            l2l.data.transforms.LoadData(meta_train_dataset),
            l2l.data.transforms.RemapLabels(meta_train_dataset),
        ]
        meta_train_task_dataset = l2l.data.TaskDataset(meta_train_dataset, task_transforms=task_transforms,
                                                       num_tasks=20000)

        X_valid, Y_valid = cls.get_X_and_Y(validation_dataset, embedding_model, max_seq_len)
        meta_test_dataset = l2l.data.MetaDataset(FewShotTextDataset(X_valid, Y_valid))

        task_transforms = [
            l2l.data.transforms.NWays(meta_test_dataset, 2),  # Binary Classification
            l2l.data.transforms.KShots(meta_test_dataset, 2 * shots),  # Number of examples
            l2l.data.transforms.LoadData(meta_test_dataset),
            l2l.data.transforms.RemapLabels(meta_test_dataset),
        ]
        meta_test_task_dataset = l2l.data.TaskDataset(meta_test_dataset, task_transforms=task_transforms,
                                                      num_tasks=20000)

        def accuracy(predictions, targets):
            predictions = predictions.argmax(dim=1).view(targets.shape)
            return (predictions == targets).sum().float() / targets.size(0)

        def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
            data, labels = batch
            data, labels = data.to(device), labels.to(device)

            adaptation_indices = np.zeros(data.size(0), dtype=bool)
            # Separate data into adaptation/evalutation sets
            adaptation_indices[np.arange(shots * ways) * 2] = True
            evaluation_indices = torch.from_numpy(~adaptation_indices)
            adaptation_indices = torch.from_numpy(adaptation_indices)
            adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
            evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

            # Adapt the model
            for step in range(adaptation_steps):
                train_error = loss(learner(adaptation_data), adaptation_labels.float().view(-1, 1))
                learner.adapt(train_error)

            # Evaluate the adapted model
            predictions = learner(evaluation_data)
            valid_error = loss(predictions, evaluation_labels.float().view(-1, 1))
            valid_accuracy = accuracy(predictions, evaluation_labels)
            return valid_error, valid_accuracy

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = torch.device('cpu')
        if cuda:
            torch.cuda.manual_seed(seed)
            device = torch.device('cuda')

        maml = l2l.algorithms.MAML(model.to(device), lr=fast_lr, first_order=False)
        opt = optim.Adam(maml.parameters(), meta_lr)
        loss = nn.BCELoss()

        for iteration in range(num_iterations):
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone()
                batch = meta_train_task_dataset.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                batch = meta_test_task_dataset.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   shots,
                                                                   ways,
                                                                   device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = meta_test_task_dataset.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
        print('Meta Test Error', meta_test_error / meta_batch_size)
        print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

        return cls(model, embedding_model)

    def predict(self, dataset: ClassificationDataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')

        X, _ = self.get_X_and_Y(dataset, self.embedding_model, max_seq_len)

        # Move the model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        # Ensure the model is in evaluation mode
        self.model.eval()

        # Create a DataLoader for the dataset
        data_loader = DataLoader(X, batch_size=metadata.get('batch_size', 32))

        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to the same device as the model
                batch = batch.to(device)

                # Forward pass through the model
                outputs = self.model(batch)

                # Convert outputs to probabilities using sigmoid since it's a binary classification
                probs = torch.sigmoid(outputs)

                # Convert these probabilities to binary predictions (0 or 1)
                batch_predictions = (probs > 0.5).int()

                # Move predictions to the CPU and convert to numpy array
                predictions.append(batch_predictions.cpu().numpy())

        # Flatten the batched predictions and return
        return np.concatenate(predictions, axis=0)

    @classmethod
    def get_X_and_Y(cls, classification_dataset, embedding_model, max_seq_len):
        codes, labels = classification_dataset.get_texts(), classification_dataset.get_labels()

        X = embedding_model.text_to_indexes(codes)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)
        Y = np.array([1 if label == True else 0 for label in labels])
        return X, Y

