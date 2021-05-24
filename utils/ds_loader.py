from abc import abstractmethod

class DatasetLoader:

    def __init__(self):
        self.num_classes = None
        self.classes = None
        self.train_ds = None
        self.test_ds = None


    @abstractmethod
    def init_dummy_features(self):
        pass


