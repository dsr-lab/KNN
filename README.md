# KNN
### Introduction
This is the very first project with the goal of solving some computer vision tasks by means of machine learning and deep learning techniques. 
In particular, the goal of this tiny repository is to solve the classification problem by means of the knn algorithm

### Dataset
The dataset used for training this classifier is CIFAR-10

### Usage
The usage is very straightforward. 

Just clone the repository and launch it on your Python environment!

In the *config.py* you can find a few constants:
* **MODEL_BEST_P** and **MODEL_BEST_K**: the best hyper-parameters that I've found so far during the training.
* **MODEL_RUN_ON_DUMMY_DATASET**: flag used for running the model on a small dataset, in order to asses the correctess of the model
* **MODEL_LOOK_FOR_BEST_HPARAMS**: set this flag to True in order to repeat the training
