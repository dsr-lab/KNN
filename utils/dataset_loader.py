import torchvision
import torchvision.transforms as transforms

class DatasetLoader:

    def __init__(self):
        self.property = 3

    def test_function(self):
        print("aaaaa: {}", self.property)

    def get_cifar10_dataset(self, root="data"):
        train_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(),
                                                download=True)
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor())

        return train_ds, test_ds


