import os

import torch
from utils import *
from knn.config import *
import torch.nn.functional as F
from timeit import default_timer as timer


class LinearClassifier:
    def __init__(self, x_train, y_train, x_validation, y_validation, n_classes, sigma=0.001):
        self.W = torch.normal(0, sigma, (x_train.shape[1], n_classes))
        self.b = torch.zeros((n_classes))

        self.W.requires_grad_(True)
        self.b.requires_grad_(True)

        self.x_train = x_train
        self.y_train = y_train

        self.x_validation = x_validation
        self.y_validation = y_validation

    def set_params(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        return torch.matmul(x, self.W) + self.b

    def train_loop(self, epochs, learning_rate, verbose=True):

        best_valid_accuracy = 0
        best_params = [self.W, self.b]
        best_epoch = -1

        for e in range(epochs):
            scores = self.forward(self.x_train)
            loss = F.cross_entropy(scores, self.y_train)
            loss.backward()

            predictions = torch.argmax(scores, 1)
            acc = accuracy(predictions, self.y_train)

            with torch.no_grad():
                self.W -= learning_rate * self.W.grad
                self.b -= learning_rate * self.b.grad

                self.W.grad.zero_()
                self.b.grad.zero_()

                valid_scores = self.forward(self.x_validation)
                valid_loss = F.cross_entropy(valid_scores, self.y_validation)

                valid_predictions = torch.argmax(valid_scores, 1)
                valid_acc = accuracy(valid_predictions, self.y_validation)

                if valid_acc > best_valid_accuracy:
                    best_valid_accuracy = valid_acc
                    best_params = {"W": self.W, "b": self.b}
                    best_epoch = e

            if epochs % 50 == 0 and verbose:
                print(f"Epoch {e}: train loss {loss:.3f}\ttrain accuracy {acc:.3f}\t"
                      f"validation loss {valid_loss:.3f}\tvalidation accuracy {valid_acc:.3f}")

        return best_valid_accuracy, best_params, best_epoch


def model_selection(linear_classifier: LinearClassifier,
                    learning_rates=[1e-6 * 10 ** i for i in range(5)],
                    epochs=1000):
    best_validation_accuracy = 0
    best_parameters = []
    best_hyper_parameters = []

    for learning_rate in learning_rates:
        current_valid_accuracy, current_best_params, current_best_epoch = \
            linear_classifier.train_loop(epochs, learning_rate, verbose=False)

        if current_valid_accuracy > best_validation_accuracy:
            best_validation_accuracy = current_valid_accuracy
            best_parameters = current_best_params
            best_hyper_parameters = {"learning_rate": learning_rate, "epoch": current_best_epoch}

            print(f"Improved result: "
                  f"acc {best_validation_accuracy:.3f}, "
                  f"lr {learning_rate}, "
                  f"epoch {current_best_epoch} ")

    return best_parameters, best_hyper_parameters


def run_model(ds: Cifar10, weights_path: os.path):

    linear_classifier = LinearClassifier(
        ds.train_features, ds.train_labels,
        ds.validation_features, ds.validation_labels,
        ds.num_classes)

    # Look for the best model
    if MODEL_LOOK_FOR_BEST_HPARAMS:
        lrs = [1e-3, 3.3e-3, 1e-2, 3.3e-2]
        start = timer()
        best_params, best_hyper_params = model_selection(linear_classifier, learning_rates=lrs, epochs=1000)
        end = timer()
        print(f"Elapsed time (s): {end - start:.3f}")
        print(f"best lr {best_hyper_params.get('learning_rate')}, best epoch {best_hyper_params.get('epoch')}")

        torch.save(best_params, os.path.join(weights_path, "best_params.th"))
        torch.save(best_hyper_params, os.path.join(weights_path, "best_hyper_params.th"))
    else:
        best_params = torch.load(os.path.join(weights_path, "best_params.th"))
        best_hyper_params = torch.load(os.path.join(weights_path, "best_hyper_params.th"))


    # Train the model
    linear_classifier.set_params(best_params.get("W"), best_params.get("b"))
    best_valid_accuracy, best_params, best_epoch = \
        linear_classifier.train_loop(
            best_hyper_params.get("epoch"),
            best_hyper_params.get("learning_rate"),
            verbose=False)

    # Run on test set
    start = timer()
    linear_classifier.set_params(best_params.get("W"), best_params.get("b"))
    test_scores = linear_classifier.forward(ds.test_features)
    end = timer()

    predictions = torch.argmax(test_scores, 1)
    test_acc = accuracy(predictions, ds.test_labels)
    print(f"Elapsed time (s): {end - start:.3f}")
    print(f"Accuracy on full test set {test_acc:.3f}")

    return best_params


def show_templates(best_param, classes):

    best_templates = best_param.get('W').detach().clone()
    # Normalize between 0/1 and reshape
    best_templates -= best_templates.min(0, keepdim=True)[0]
    best_templates /= best_templates.max(0, keepdim=True)[0]
    best_templates = best_templates.reshape(3, 32, 32, 10)

    n_classes = len(classes)
    for i in range(n_classes):
        plt.subplot(2, n_classes/2, i+1)
        plot.show_image(best_templates[:, :, :, i])
        plt.title(classes[i])
    plt.subplots_adjust(hspace=0.02)
    plt.show(block=True)


def main():
    ds = Cifar10()

    if MODEL_RUN_ON_DUMMY_DATASET:
        ds.init_dummy_features()
    else:
        ds.init_features()

    current_path = os.path.abspath(os.getcwd()+'/model')

    if not os.path.exists(current_path):
        os.makedirs(current_path)

    best_param = run_model(ds, current_path)
    show_templates(best_param, ds.classes)


if __name__ == "__main__":
    main()
