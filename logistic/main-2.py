import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mkpicdata import make_image_set

import time

def phi(x):
    return np.append(x, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

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
    e = 0.001 # learning rate
    n = 100 # number of train
    train_size = 10000
    test_size = 1000

    study_sample = make_image_set(train_size, cls=(0, 1))
    test_sample = make_image_set(test_size, cls=(0, 1))

    w = np.random.randn(28*28*3+1)
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
