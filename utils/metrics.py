def accuracy(predictions, labels):

    n_samples = labels.shape[0]
    a = (labels == predictions)
    n_correct = a.sum().item()

    return n_correct/n_samples
