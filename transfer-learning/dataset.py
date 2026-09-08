import os
import random
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch.utils.data import Dataset


def make_dir(file_path: str | Path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def separate_train_test(dataset_path: str | Path, train_path: str | Path, test_path: str | Path):
    """
    Separates CUB-200-2011 images into training and test sets.
    First 15 images per class go to test, the rest to train.
    """
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith("."):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith("."):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1


class NoisyLabelDataset(Dataset):
    def __init__(self, original_dataset: Dataset, num_classes: int, noise_percentage: float = 0.1):
        self.original_dataset = original_dataset
        self.noise_percentage = noise_percentage
        self.num_samples = len(original_dataset)
        self.num_classes = num_classes
        num_errors = int(noise_percentage * self.num_samples)
        self.error_indices = random.sample(range(self.num_samples), num_errors)
        self.errors = np.random.randint(0, self.num_classes, (num_errors,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        data, label = self.original_dataset[index]
        if index in self.error_indices:
            label = self.errors[self.error_indices.index(index)]
        return data, label
