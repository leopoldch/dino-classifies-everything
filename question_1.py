import os
import random
from pathlib import Path
from shutil import copyfile

import numpy as np
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T
from transformers import AutoModelForImageClassification


def make_dir(file_path: str | Path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def separate_train_test(dataset_path: str | Path, train_path: str | Path, test_path: str | Path):
    """
    This function separates the images of CUB200 into a training and test set.
    :param dataset_path: Root path where the images of CUB200 are located.
    :param train_path: Output path to save the training set.
    :param test_path: Output path to save the test set.
    :return:
    """
    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1


class NoisyLabelDataset(Dataset):
    def __init__(self, original_dataset: Dataset, num_classes: int, noise_percentage: float = 0.1):
        """
        Modifies the labels of a dataset to introduce noise.
        :param original_dataset: Original dataset to wrap.
        :param num_classes: Number of classes in the dataset.
        :param noise_percentage: Fraction of labels to introduce noise (e.g., 0.05 for 5% noise).
        """
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


if __name__ == '__main__':
    data_root = Path('data/cub200')
    if not os.path.exists(data_root):
        separate_train_test('data/CUB_200_2011/CUB_200_2011/images', data_root / 'train', data_root / 'test')

    # TODO Faire les transformations
    train_transform = T.Compose([])
    test_transform = T.Compose([])

    train_dataset = torchvision.datasets.ImageFolder(f'{data_root}/train', transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(f'{data_root}/test', transform=test_transform)


    # Pour ajouter du bruit sur les étiquettes
    num_classes = len(train_dataset.classes)
    noisy_train_dataset = NoisyLabelDataset(train_dataset, num_classes=num_classes, noise_percentage=0.1)
    
    # Pour utiliser un modèle pré-entraîné DINOv2 (voir : https://huggingface.co/facebook/dinov2-small et https://huggingface.co/docs/transformers/model_doc/dinov2#transformers.Dinov2ForImageClassification)
    model = AutoModelForImageClassification.from_pretrained(
        'facebook/dinov2-small',
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )