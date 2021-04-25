import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from utils import DatasetLoader


class Cifar10(DatasetLoader):

    def __init__(self):
        self.train_features = None
        self.train_labels = None

        self.test_features = None
        self.test_labels = None

        self.validation_features = None
        self.validation_labels = None

        self.__create()

    def init_dummy_features(self):
        splitted_datasets = torch.utils.data.random_split(self.train_ds, [45000, 5000])
        train_subds = splitted_datasets[0]
        valid_subds = splitted_datasets[1]

        self.train_features, self.train_labels = self.__subset_to_matrix(train_subds, 500)
        self.validation_features, self.validation_labels = self.__subset_to_matrix(valid_subds, 100)
        self.test_features, self.test_labels = self.__dataset_to_matrix(self.test_ds, 100)

    def init_features(self):
        splitted_datasets = torch.utils.data.random_split(self.train_ds, [45000, 5000])
        train_subds = splitted_datasets[0]
        valid_subds = splitted_datasets[1]

        self.train_features, self.train_labels = self.__subset_to_matrix(train_subds)
        self.validation_features, self.validation_labels = self.__subset_to_matrix(valid_subds)
        self.test_features, self.test_labels = self.__dataset_to_matrix(self.test_ds)

    def __create(self, root="data"):
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse',
                        'ship', 'truck')

        self.train_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(),
                                                     download=True)
        self.test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor())

    def __subset_to_matrix(self, dset, limit=None):

        # shape: (n_dataset, 3, 32, 32)
        x = torch.tensor(dset.dataset.data[dset.indices, ], dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
        # shape: (n_dataset, 3072) flat operation
        x = x.reshape((x.shape[0], -1))

        # Convert targets to a tensor by considering only the correct indeces
        # shape: (n_dataset, 1)
        y = torch.tensor(np.array(dset.dataset.targets)[dset.indices], dtype=torch.int64)

        if limit is not None:
            if limit <= 0 or limit >= x.shape[0]:
                raise ValueError("Invalid limit, cannot be negative or greater than the input dataset size")

            x = x[:limit].clone()
            y = y[:limit].clone()

        return x, y

    def __dataset_to_matrix(self, dset, limit=None):
        x = torch.tensor(dset.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
        x = x.reshape((x.shape[0], -1))
        y = torch.tensor(dset.targets, dtype=torch.int64)
        if limit is not None:
            if limit <= 0 or limit > x.shape[0]:
                raise ValueError("Invalid limit, cannot be negative or greater than the input dataset size")
            x = x[:limit].clone()
            y = y[:limit].clone()
        return x, y
