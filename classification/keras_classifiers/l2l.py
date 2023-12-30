import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class FewShotTextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.int64), self.labels[idx]


class BinaryClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super(BinaryClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float))
        self.embedding.weight.requires_grad = True  # Set to False if you want to freeze the embedding layer
        self.conv1 = nn.Conv1d(embedding_dim, 100, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # Change the shape for Conv1d
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
