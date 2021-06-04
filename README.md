# KNN and Linear Classifier
### Introduction
This is the very first project of a larger set, which has has the goal of solving some computer vision tasks by means of machine learning and deep learning techniques. 

In particular, the goal of this tiny repository is to solve the classification problem by means of knn classifier and linear classifier.

### Dataset
The dataset used for training this classifier is CIFAR-10

### Usage
The usage is very straightforward: just clone the repository and launch it on your Python environment!

In the *config.py* you can find a few configuration parameters:
* **MODEL_BEST_P** and **MODEL_BEST_K**: the best hyper-parameters that I've found so far during the training (used only for KNN classifier)
* **MODEL_RUN_ON_DUMMY_DATASET**: flag used for running the model on a small dataset, in order to asses the correctess of the model
* **MODEL_NEED_TRAINING**: set this flag to True in order to perform the training

### Observations
The final accuracy for both knn and linear classifier is similar, and very low: using directly pixels values as features for trainining a model is not a very good choice. This is one of the reasons why neural networks work much better for doing image classification.
