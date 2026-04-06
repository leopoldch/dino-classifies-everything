import random

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Subset

from borealtc import BorealTC, SlidingWindowDataset


def split_train_test(dataset):
    classes = dataset.classes
    test_indices = []
    for c in classes:
        indices = [i for i, d in enumerate(dataset) if d.class_name == c]
        test_indices.append(random.choice(indices))
    train_indices = list(set(range(len(dataset))) - set(test_indices))
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


if __name__ == '__main__':
    window_size = 170
    step_size = 10

    dataset = BorealTC('data/borealtc')
    class_to_idx = dataset.class_to_idx
    columns = dataset.columns

    train_dataset, test_dataset = split_train_test(dataset)
    train_dataset = SlidingWindowDataset(train_dataset, window_size, step_size)
    test_dataset = SlidingWindowDataset(test_dataset, window_size, step_size)

    for i in np.random.randint(len(train_dataset), size=10):
        sample = train_dataset[i]
        window = sample['window']
        time = np.linspace(0, window_size / 100, window_size)

        plt.title(sample['class_name'])
        plt.xlabel('Time (s)')
        for j in range(len(columns)):
            plt.plot(time, window[:, j], label=columns[j])
        plt.legend()
        plt.show()
