from sklearn.utils import compute_class_weight
import random

import learn2learn as l2l
import numpy as np
import tensorflow as tf
import torch
from keras import layers
from keras.src.utils import pad_sequences
from sklearn.utils import compute_class_weight
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm

from classification.models import ClassifierModel, ClassificationDataset
from classification.torch_classifier.deep_line_dp import HierarchicalAttentionNetwork
from classification.torch_classifier.l2l import BinaryClassifier, FewShotTextDataset


class CNN(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, keep_probab, vocab_size, embedding_dim):
        super(CNN, self).__init__()
        '''
        Arguments
        ---------
        batch_size : Size of each batch
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        keep_probab : Probability of retaining an activation node during dropout operation
        vocab_size : Size of the vocabulary containing unique words
        '''
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.conv = nn.Conv2d(1, 100, (5, embedding_dim))

        self.dropout = nn.Dropout(keep_probab)
        self.fc = nn.Linear(100, 1)

        self.sig = nn.Sigmoid()

    def conv_block(self, input, conv_layer):
        conv_out = F.relu(conv_layer(input))
        conv_out = torch.squeeze(conv_out, -1)
        max_out = F.max_pool1d(conv_out, conv_out.size()[2])

        return max_out

    def forward(self, input_tensor):
        '''
        Parameters
        ----------
        input_tensor: input_tensor of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for pos & neg class.

        '''
        # input.size() = (batch_size, num_seq, embedding_length)
        input = self.word_embeddings(input_tensor.type(torch.LongTensor))
        # input = self.word_embeddings(input_tensor.type(torch.LongTensor).cuda())

        # input.size() = (batch_size, 1, num_seq, embedding_length)
        input = input.unsqueeze(1)

        conv_out = F.relu(nn.Conv2d(1, 100, (5, self.embedding_dim))(input))
        conv_out = torch.squeeze(conv_out, -1)
        max_out = F.max_pool1d(conv_out, conv_out.size()[2])

        max_out = max_out.view(max_out.size(0), -1)

        fc_in = nn.Dropout(0.5)(max_out)

        logits = nn.Linear(100, 1)(fc_in)
        sig_out = nn.Sigmoid()(logits)

        return sig_out


class TorchClassifier(ClassifierModel):
    def __init__(self, net, embedding_model):
        self.net = net
        self.embedding_model = embedding_model

    @classmethod
    def from_training(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        dataset_name = metadata.get('dataset_name')

        n_filters = 100
        lr = 0.001
        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        train_code, train_label = train_dataset.get_texts(), train_dataset.get_labels()
        X_train = embedding_model.text_to_indexes(train_code)
        X_train = pad_sequences(X_train, padding='post', maxlen=max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]

        Y_train = np.array([1 if label == True else 0 for label in train_label])

        train_tensor_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(np.array(Y_train).astype(int)))
        train_dl = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size, drop_last=True)

        valid_code, valid_label = validation_dataset.get_texts(), validation_dataset.get_labels()
        X_valid = embedding_model.text_to_indexes(valid_code)
        X_valid = pad_sequences(X_valid, padding='post', maxlen=max_seq_len)
        Y_valid = np.array([1 if label == True else 0 for label in valid_label])
        valid_tensor_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(np.array(Y_valid).astype(int)))
        valid_dl = DataLoader(valid_tensor_data, shuffle=True, batch_size=batch_size, drop_last=True)

        net = CNN(batch_size, 1, n_filters, 0.5, vocab_size, embedding_dim)

        # net = net.cuda()

        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        criterion = nn.BCELoss()

        current_checkpoint_num = 1

        train_loss_all_epochs = []
        val_loss_all_epochs = []

        clip = 5

        print('training model of', dataset_name)

        for e in tqdm(range(current_checkpoint_num, epochs + 1)):
            train_losses = []
            val_losses = []

            net.train()

            # batch loop
            for inputs, labels in train_dl:
                # inputs, labels = inputs.cuda(), labels.cuda()
                net.zero_grad()

                # get the output from the model
                output = net(inputs)

                # calculate the loss and perform backprop
                loss = criterion(output, labels.reshape(-1, 1).float())
                train_losses.append(loss.item())

                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()

            train_loss_all_epochs.append(np.mean(train_losses))

            with torch.no_grad():

                net.eval()

                for inputs, labels in valid_dl:
                    # inputs, labels = inputs.cuda(), labels.cuda()
                    output = net(inputs)

                    val_loss = criterion(output, labels.reshape(batch_size, 1).float())

                    val_losses.append(val_loss.item())

                val_loss_all_epochs.append(np.mean(val_losses))
        print('finished training model of', dataset_name)
        return cls(net, embedding_model)

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')
        batch_size = metadata.get('batch_size')

        net = self.net
        # net = net.cuda()
        test_df = dataset

        code, labels = test_df.get_texts(), test_df.get_labels()
        X = self.embedding_model.text_to_indexes(code)
        X = pad_sequences(X, padding='post', maxlen=max_seq_len)

        Y = np.array([1 if label == True else 0 for label in labels])
        tensor_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(np.array(Y).astype(int)))
        dl = DataLoader(tensor_data, shuffle=False, batch_size=batch_size)

        outputs = []
        for inputs, labels in dl:
            output = net(inputs)
            outputs.extend(output.flatten().tolist())
        return outputs


class TorchHANClassifier(ClassifierModel):
    def __init__(self, model, embedding_model):
        self.model = model
        self.embedding_model = embedding_model

    max_sent_num = 1000

    @classmethod
    def get_loss_weight(cls, labels, weight_dict):
        '''
            input
                labels: a PyTorch tensor that contains labels
            output
                weight_tensor: a PyTorch tensor that contains weight of defect/clean class
        '''
        label_list = labels.cpu().numpy().squeeze().tolist()
        weight_list = []

        for lab in label_list:
            if lab == 0:
                weight_list.append(weight_dict['clean'])
            else:
                weight_list.append(weight_dict['defect'])

        weight_tensor = torch.tensor(weight_list).reshape(-1, 1)
        return weight_tensor

    @classmethod
    def from_training(cls, train_dataset, validation_dataset=None, metadata=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        max_grad_norm = 5
        word_gru_hidden_dim = 100
        sent_gru_hidden_dim = 100
        word_gru_num_layers = 100
        sent_gru_num_layers = 100
        word_att_dim = 64
        sent_att_dim = 64
        lr = 0.001
        use_layer_norm = True
        dropout = 0.5
        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        train_dataset = train_dataset
        valid_df = validation_dataset

        train_code, train_label = train_dataset.get_texts(), train_dataset.get_labels()
        valid_code, valid_label = valid_df.get_texts(), valid_df.get_labels()

        x_train_vec = cls.get_codes_3d(embedding_model, max_seq_len, train_code)
        x_valid_vec = cls.get_codes_3d(embedding_model, max_seq_len, valid_code)

        sample_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)

        Y_train = np.array([1 if label == True else 0 for label in train_label])
        train_tensor_data = TensorDataset(torch.from_numpy(x_train_vec),
                                          torch.from_numpy(np.array(Y_train).astype(int)))
        train_dl = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size, drop_last=True)

        Y_valid = np.array([1 if label == True else 0 for label in valid_label])
        valid_tensor_data = TensorDataset(torch.from_numpy(x_valid_vec),
                                          torch.from_numpy(np.array(Y_valid).astype(int)))
        valid_dl = DataLoader(valid_tensor_data, shuffle=True, batch_size=batch_size, drop_last=True)

        weight_dict = {}
        weight_dict['defect'] = np.max(sample_weights)
        weight_dict['clean'] = np.min(sample_weights)

        model = HierarchicalAttentionNetwork(
            vocab_size=vocab_size,
            embed_dim=embedding_dim,
            word_gru_hidden_dim=word_gru_hidden_dim,
            sent_gru_hidden_dim=sent_gru_hidden_dim,
            word_gru_num_layers=word_gru_num_layers,
            sent_gru_num_layers=sent_gru_num_layers,
            word_att_dim=word_att_dim,
            sent_att_dim=sent_att_dim,
            use_layer_norm=use_layer_norm,
            dropout=dropout
        )

        model = model
        model.sent_attention.word_attention.freeze_embeddings(False)

        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        criterion = nn.BCELoss()

        train_loss_all_epochs = []
        val_loss_all_epochs = []

        for epoch in tqdm(range(0, epochs + 1)):
            train_losses = []
            val_losses = []

            model.train()

            for inputs, labels in train_dl:
                inputs_cuda, labels_cuda = inputs, labels
                output, _, __, ___ = model(inputs_cuda)

                weight_tensor = cls.get_loss_weight(labels, weight_dict)

                criterion.weight = weight_tensor

                loss = criterion(output, labels_cuda.reshape(batch_size, 1))

                train_losses.append(loss.item())

                # torch.cuda.empty_cache()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

                # torch.cuda.empty_cache()

            train_loss_all_epochs.append(np.mean(train_losses))

            with torch.no_grad():

                criterion.weight = None
                model.eval()

                for inputs, labels in valid_dl:
                    inputs, labels = inputs, labels
                    output, _, __, ___ = model(inputs)

                    val_loss = criterion(output, labels.reshape(batch_size, 1))

                    val_losses.append(val_loss.item())

                val_loss_all_epochs.append(np.mean(val_losses))

    @classmethod
    def get_codes_3d(cls, embedding_model, max_seq_len, codes):
        codes_3d = np.zeros((len(codes), TorchHANClassifier.max_sent_num, max_seq_len), dtype='int32')
        for file_idx, file_code in enumerate(codes):
            for line_idx, line in enumerate(file_code.splitlines()):
                if line_idx >= TorchHANClassifier.max_sent_num:
                    continue

                X = embedding_model.text_to_indexes([line])[0]
                X = pad_sequences([X], padding='post', maxlen=max_seq_len)
                codes_3d[file_idx, line_idx, :] = X

        return codes_3d

    def predict(self, dataset, metadata=None):
        max_seq_len = metadata.get('max_seq_len')
        batch_size = metadata.get('batch_size')

        model = self.model
        model.sent_attention.word_attention.freeze_embeddings(True)
        # model = model.cuda()
        model.eval()

        test_df = dataset

        code, labels = test_df.get_texts(), test_df.get_labels()
        x_vec = self.get_codes_3d(self.embedding_model, max_seq_len, code)
        Y = np.array([1 if label == True else 0 for label in labels])

        tensor_data = TensorDataset(torch.from_numpy(x_vec),
                                    torch.from_numpy(np.array(Y).astype(int)))
        dl = DataLoader(tensor_data, shuffle=False, batch_size=batch_size)

        outputs = []
        for inputs, labels in dl:
            with torch.no_grad():
                output, word_att_weights, line_att_weight, _ = model(inputs)
                output = model(inputs)
                outputs.extend(output.flatten().tolist())
        return outputs


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
