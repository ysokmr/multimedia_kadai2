import numpy as np

class Data:
    def __init__(self, cls, ave=0, sig=1, dim=2, num_train=1000, num_test=100):
        self.cls = cls
        self.train = np.empty((num_train, dim))
        self.test = np.empty((num_test, dim))
        self.ave = ave
        self.sig = sig
        self.dim = dim
        self.num_train = num_train
        self.num_test = num_test

    def _genPoint(self):
        return (np.random.normal(self.ave, self.sig, (self.dim)), self.cls)

    def genTrain(self):
        train = []
        for _ in range(self.num_train):
            train.append(self._genPoint())
        self.train = np.array(train)
        return self.train

    def genTest(self):
        test = []
        for _ in range(self.num_test):
            test.append(self._genPoint())
        self.test = np.array(test)
        return self.test

    def genDataset(self):
        self.genTrain()
        self.genTest()
        return (self.train, self.test)


