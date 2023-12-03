import numpy as np
from keras.src.utils import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from classification.models import ClassifierModel
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import functional as F


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
    def train(cls, source_code_df, dataset_name, metadata=None, validation_source_code_df=None):
        embedding_model = metadata.get('embedding_model')
        batch_size = metadata.get('batch_size')
        epochs = metadata.get('epochs')
        max_seq_len = metadata.get('max_seq_len')
        embedding_matrix = metadata.get('embedding_matrix')
        n_filters = 100
        lr = 0.001
        if embedding_model is not None:
            vocab_size = embedding_model.get_vocab_size()
            embedding_dim = embedding_model.get_embedding_dim()
        else:
            vocab_size = metadata.get('vocab_size')
            embedding_dim = metadata.get('embedding_dim')

        train_df = source_code_df
        valid_df = validation_source_code_df

        train_code, train_label = train_df['text'], train_df['label']
        X_train = embedding_model.text_to_indexes(train_code)
        X_train = pad_sequences(X_train, padding='post', maxlen=max_seq_len)
        if max_seq_len is None:
            max_seq_len = X_train.shape[1]

        Y_train = np.array([1 if label == True else 0 for label in train_label])

        train_tensor_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(np.array(Y_train).astype(int)))
        train_dl = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size, drop_last=True)

        valid_code, valid_label = valid_df['text'], valid_df['label']
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

    def predict(self, df, metadata=None):
        max_seq_len = metadata.get('max_seq_len')
        batch_size = metadata.get('batch_size')

        net = self.net
        # net = net.cuda()
        test_df = df

        code, labels = test_df['text'], test_df['label']
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
