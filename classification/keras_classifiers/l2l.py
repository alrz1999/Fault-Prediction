import learn2learn as l2l
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FewShotTextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float), self.labels[idx]


class BinaryClassifier(nn.Module):
    def __init__(self, embedding_size):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(embedding_size, 2)  # Adjust the size according to your needs

    def forward(self, x):
        return self.fc(x)

model = BinaryClassifier(embedding_size=YOUR_EMBEDDING_SIZE)



maml = l2l.algorithms.MAML(model, lr=0.01, first_order=False)
optimizer = optim.Adam(maml.parameters(), lr=0.005)
loss_func = nn.CrossEntropyLoss()

num_iterations = 500
num_tasks = 20
num_adaptation_steps = 1

for iteration in range(num_iterations):
    iteration_error = 0.0
    for task in range(num_tasks):
        # Sample a task
        learner = maml.clone()
        task_description, train_loader, _ = meta_train_dataset.sample(task_transforms=[
            l2l.data.transforms.NWays(meta_train_dataset, ways=2),  # Binary Classification
            l2l.data.transforms.KShots(meta_train_dataset, 2 * shots),  # Number of examples
            l2l.data.transforms.LoadData(meta_train_dataset),
            l2l.data.transforms.RemapLabels(meta_train_dataset),
        ])

        # Adapt the model
        for adaptation_step, (X_train, Y_train) in enumerate(train_loader):
            train_error = loss_func(learner(X_train), Y_train)
            learner.adapt(train_error)
            if adaptation_step >= num_adaptation_steps - 1:
                break

        # Calculate loss on the same task to update meta-learner
        _, _, test_loader = task_description
        for X_test, Y_test in test_loader:
            test_error = loss_func(learner(X_test), Y_test)
            iteration_error += test_error

    # Update the meta-learner
    optimizer.zero_grad()
    iteration_error /= (num_tasks * num_adaptation_steps)
    iteration_error.backward()
    optimizer.step()

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Loss {iteration_error.item()}")

# Evaluate the model on new tasks for meta-testing
total_accuracy = 0.0
for task in range(num_tasks):
    # Sample a new task
    learner = maml.clone()
    task_description, _, test_loader = meta_test_dataset.sample(task_transforms=[
        l2l.data.transforms.NWays(meta_test_dataset, ways=2),
        l2l.data.transforms.KShots(meta_test_dataset, shots),
        l2l.data.transforms.LoadData(meta_test_dataset),
        l2l.data.transforms.RemapLabels(meta_test_dataset),
    ])

    # Evaluate the learner on the new task
    correct = 0
    total = 0
    for X_test, Y_test in test_loader:
        with torch.no_grad():
            predictions = learner(X_test).argmax(dim=1)
            correct += (predictions == Y_test).sum().item()
            total += Y_test.size(0)

    total_accuracy += correct / total

print(f"Average accuracy over {num_tasks} tasks: {total_accuracy / num_tasks * 100}%")
