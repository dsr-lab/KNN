import torch
import torch.nn.functional as F

from utils import *


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
