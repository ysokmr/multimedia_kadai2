import math
import random
import numpy as np
import matplotlib.pyplot as plt
from data_generator import Data

import time

def phi(x):
    return np.append(x, 1)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-1 * x))

def judge(predict):
    if predict >= 0.5:
        return 1
    else:
        return 0

def train_param(study_sample :list, w :list, eta=0.001):
    s = 0
    for data in study_sample:
        x, t = data
        feature = phi(x)
        predict = sigmoid(np.inner(w, feature))
        s += (predict - t) * feature
    return w - (eta*s)

def get_correct_rate(test_sample :list, w):
    correct = 0
    count = 0
    for data in test_sample:
        x, t = data
        feature = phi(x)
        predict = sigmoid(np.inner(w, feature))
        if judge(predict) == t:
            correct += 1
        count += 1
    return correct / count

if __name__ == "__main__":
    e = 0.001
    n = 100
    train_num = 2000
    test_num = 1000
    dim = 2
    ave1 = (-2, 1)
    sig1 = (1, 2)
    ave2 = (0, 1)
    sig2 = (1, 2)

    data1 = Data(0, ave1, sig1, dim, train_num, test_num)
    data1.genDataset()
    data2 = Data(1, ave2, sig2, dim, train_num, test_num)
    data2.genDataset()

    study_sample = np.vstack((data1.train, data2.train))
    test_sample = np.vstack((data1.test, data2.test))

    w = np.random.randn(dim+1)
    print(get_correct_rate(test_sample, w))
    t0 = time.perf_counter()
    for i in range(n):
        if i % (n//10) == 0:
            print(i)
            print(w)
            print(get_correct_rate(test_sample, w))
        w = train_param(study_sample, w, e)
    t1 = time.perf_counter()

    print(n)
    print(w)
    print(get_correct_rate(test_sample, w))
    print(t1 - t0)

    # show graph
    x1 = []
    y1 = []
    x2 = []
    y2 = [] 
    x = np.linspace(-5, 5, 200)
    y = (0.5 - w[2] - x*w[0]) / w[1]
    for point in test_sample:
        if point[1] == 1:
            x1.append(point[0][0])
            y1.append(point[0][1])
        else:
            x2.append(point[0][0])
            y2.append(point[0][1])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.ylim((-5, 5))
    ax.scatter(x1,y1, c='red')
    ax.scatter(x2,y2, c='blue')
    ax.plot(x, y, color='black') #...3

    plt.savefig('logistic.png')
    plt.show()
