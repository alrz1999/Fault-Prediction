import pandas as pd
from sklearn.model_selection import train_test_split
import random


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
