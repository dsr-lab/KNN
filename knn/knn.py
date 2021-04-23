from utils import DatasetLoader


class KNN:
    def __init__(self):
        ds = DatasetLoader()
        self.train_ds, self.test_ds = ds.get_cifar10_dataset(root="data")

    def test_knn(self):
        print(self.train_ds)


def main():
    knn_model = KNN()
    knn_model.test_knn()


if __name__ == "__main__":
    main()
