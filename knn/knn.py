from utils import *

class KNN:
    def __init__(self):
        self.ds = DatasetLoader()
        self.ds.get_cifar10_dataset(root="data")

    def test_knn(self):

        print(self.ds.classes)
        plot.show_image(self.ds.train_ds[0][0])

        input("Press Enter to continue...")

        images = []
        labels = []
        for i in range(32):
            images.append(self.ds.train_ds[i][0])
            labels.append(self.ds.classes[self.ds.train_ds[i][1]])
        print(labels)

        plot.show_images_grid(images)


def main():
    knn_model = KNN()
    knn_model.test_knn()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
