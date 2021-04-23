import torchvision
import torchvision.transforms as transforms


class DatasetLoader:

    def __init__(self):
        self.num_classes = None
        self.classes = None
        self.train_ds = None
        self.test_ds = None

    def test_function(self):
        print("aaaaa: {}", self.property)

    def get_cifar10_dataset(self, root="data"):
        self.num_classes = 10
        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse',
                        'ship', 'truck')

        self.train_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=transforms.ToTensor(),
                                                     download=True)
        self.test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor())
