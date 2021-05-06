import torch
from utils import *
from knn.config import *
from timeit import default_timer as timer


class KnnClassifier:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, k=MODEL_BEST_K, p=MODEL_BEST_P):
        n_test = x_test.shape[0]
        pred = torch.zeros(n_test, dtype=torch.int64)
        knn_indices = torch.zeros((n_test, k), dtype=torch.int64)

        for test_index in range(n_test):
            # take single element to classify
            test_vector = x_test[test_index, :]
            # compute distances based on train samples and test
            dists = torch.norm((self.x_train - test_vector), p=p, dim=1)
            # get k smallest distances
            _, indices = torch.topk(dists, k, largest=False)

            # the sample with the shorter distance should be selected more frequently.
            # Therefore, use the argmax for getting the prediction
            pred[test_index] = torch.bincount(self.y_train[indices]).argmax()

            # Save indices
            knn_indices[test_index, :] = indices

        return pred, knn_indices


def model_selection(self, x, labels):
    start = timer()
    best_acc = 0
    best_p = 0
    best_k = 0

    for k in [1, 3, 5, 7, 10, 15, 30]:
        for p in [1, 2]:
            pred, knn_indices = self.predict(x, k=k, p=p)
            acc = accuracy(pred, labels)
            print("new accuracy: {}. p={}, k={}".format(acc, p, k))

            '''
            new accuracy: 0.3798. p=1, k=1
            new accuracy: 0.3544. p=2, k=1
            new accuracy: 0.3654. p=1, k=3
            new accuracy: 0.3408. p=2, k=3
            new accuracy: 0.373. p=1, k=5
            new accuracy: 0.3354. p=2, k=5
            new accuracy: 0.37. p=1, k=7
            new accuracy: 0.3426. p=2, k=7
            new accuracy: 0.3698. p=1, k=10
            new accuracy: 0.3424. p=2, k=10
            new accuracy: 0.3696. p=1, k=15
            new accuracy: 0.3382. p=2, k=15
            new accuracy: 0.3692. p=1, k=30
            new accuracy: 0.3268. p=2, k=30

            '''

            if acc > best_acc:
                best_acc = acc
                best_p = p
                best_k = k
                print("    found new best accuracy")

    end = timer()
    print(f"model selection required {end - start:.3f} seconds")
    return best_k, best_p


def run_model(ds: Cifar10, best_k=MODEL_BEST_K, best_p=MODEL_BEST_P):

    # Create the dataset merging train and validation set
    train_features = torch.cat((ds.train_features, ds.validation_features))
    train_labels = torch.cat((ds.train_labels, ds.validation_labels))

    # Create the model
    knn_model = KnnClassifier(train_features, train_labels)

    # Find best hyper-parameters
    if MODEL_LOOK_FOR_BEST_HPARAMS:
        knn_model_selection = KnnClassifier(ds.train_features, ds.train_labels)
        best_k, best_p = model_selection(knn_model_selection, ds.validation_features, ds.validation_labels)

    # Run on the test set and show the results
    pred, indices = knn_model.predict(ds.test_features, k=best_k, p=best_p)
    acc = accuracy(pred, ds.test_labels)
    print(f"accuracy: {acc}")

    show_predictions(train_features, ds.test_features, ds.test_labels, indices, pred, ds.classes)


def show_predictions(train_features, test_features, test_labels, knn_indices, predictions, classes):
    n_classes = 10
    for i in range(n_classes):
        # Tensor that contains an array. Each element of the array is the index of the current "i" class
        sample_indices = torch.nonzero(test_labels == i, as_tuple=False)
        plt.subplot(knn_indices.shape[1] + 1, n_classes, i + 1)

        img = test_features[sample_indices[0], :].reshape(3, 32, 32)
        title = classes[i] + "\n" + classes[predictions[sample_indices[0]]]
        plot.show_image(img, title)

        for j in range(knn_indices.shape[1]):
            plt.subplot(knn_indices.shape[1] + 1, n_classes, (j + 1) * n_classes + (i + 1))
            train_index = knn_indices[sample_indices[0], j]

            img = train_features[train_index].reshape(3, 32, 32)
            plot.show_image(img)
        plt.subplots_adjust(hspace=0.2)


def main():
    ds = Cifar10()

    if MODEL_RUN_ON_DUMMY_DATASET:
        ds.init_dummy_features()
    else:
        ds.init_features()

    run_model(ds)

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
