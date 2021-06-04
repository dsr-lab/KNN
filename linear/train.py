import os

import torch

from linear.linear_classifier import LinearClassifier
from utils import *
from knn.config import *
from timeit import default_timer as timer


def run_model(ds: Cifar10, weights_path: os.path):

    linear_classifier = LinearClassifier(
        ds.train_features, ds.train_labels,
        ds.validation_features, ds.validation_labels,
        ds.num_classes)

    if MODEL_NEED_TRAINING:
        best_hyper_params, best_params = find_best_hyper_parameters(linear_classifier, weights_path)
        best_params = train_model(best_hyper_params, best_params, linear_classifier, weights_path)
    else:
        best_params = load_hyper_parameters(weights_path)

    test_model(best_params, ds, linear_classifier)

    return best_params


def find_best_hyper_parameters(linear_classifier, weights_path):
    lrs = [1e-3, 3.3e-3, 1e-2, 3.3e-2]

    start = timer()
    best_params, best_hyper_params = \
        run_model_hyper_parameters_selection(linear_classifier, learning_rates=lrs, epochs=1000)
    end = timer()

    print(f"Elapsed time (s): {end - start:.3f}")
    print(f"best lr {best_hyper_params.get('learning_rate')}, best epoch {best_hyper_params.get('epoch')}")

    # torch.save(best_hyper_params, os.path.join(weights_path, "best_hyper_params.th"))

    return best_hyper_params, best_params


def run_model_hyper_parameters_selection(linear_classifier: LinearClassifier,
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


def load_hyper_parameters(weights_path):
    return torch.load(os.path.join(weights_path, "best_params.th"))


def train_model(best_hyper_params, best_params, linear_classifier, weights_path):
    linear_classifier.set_params(best_params.get("W"), best_params.get("b"))

    best_valid_accuracy, best_params, best_epoch = \
        linear_classifier.train_loop(
            best_hyper_params.get("epoch"),
            best_hyper_params.get("learning_rate"),
            verbose=False)

    torch.save(best_params, os.path.join(weights_path, "best_params.th"))
    return best_params


def test_model(best_params, ds, linear_classifier):
    start = timer()
    linear_classifier.set_params(best_params.get("W"), best_params.get("b"))
    test_scores = linear_classifier.forward(ds.test_features)
    end = timer()

    predictions = torch.argmax(test_scores, 1)
    test_acc = accuracy(predictions, ds.test_labels)

    print(f"Elapsed time (s): {end - start:.3f}")
    print(f"Accuracy on full test set {test_acc:.3f}")


def show_templates(best_param, classes):
    best_templates = best_param.get('W').detach().clone()

    # Normalize between 0/1 and reshape
    best_templates -= best_templates.min(0, keepdim=True)[0]
    best_templates /= best_templates.max(0, keepdim=True)[0]
    best_templates = best_templates.reshape(3, 32, 32, 10)

    n_classes = len(classes)
    for i in range(n_classes):
        plt.subplot(2, n_classes / 2, i + 1)
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
