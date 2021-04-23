from utils import DatasetLoader


class KNN:
    def __init__(self):
        self.ds = DatasetLoader()

    def test_knn(self):
        self.ds.test_function()



def main():
    knn_model = KNN()
    knn_model.test_knn()


if __name__ == "__main__":
    main()