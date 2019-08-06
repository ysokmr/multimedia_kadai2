import numpy as np

class KNN:
    def __init__(self, dataset, classes, distf=lambda x, y: np.linalg.norm(x-y)):
        self.dist = distf
        self.dataset = dataset
        self.classes = classes

    def getKNearest(self, p, k=3):
        np.random.shuffle(self.dataset)
        l = sorted(self.dataset, key = lambda x: self.dist(p, x[0]))
        return l[:k]

    def getClass(self, p, k=3):
        nn = self.getKNearest(p, k)
        cls_nn = list(map(lambda x: x[-1], nn))

        cls_num = list(map(lambda x: cls_nn.count(x), self.classes))
        cls_i = np.argmax(cls_num)
        return (self.classes[cls_i], cls_num[cls_i], nn)
