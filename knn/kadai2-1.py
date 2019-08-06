import numpy as np
import matplotlib.pyplot as plt
from data_generator import Data
from knn import KNN

import time

k = 31
dim = 2
ave1 = (4, 2)
sig1 = (2, 1)
ave2 = (-2, 2)
sig2 = (1, 2)
ave3 = (0, -2)
sig3 = (2, 2)
train_size = 1000
test_size = 100

data1 = Data(1, ave1, sig1, dim, train_size, test_size)
data1.genDataset()

data2 = Data(2, ave2, sig2, dim, train_size, test_size)
data2.genDataset()

data3 = Data(3, ave3, sig3, dim, train_size, test_size)
data3.genDataset()


traindata = np.vstack((data1.train, data2.train, data3.train))

for i in range(train_size):
    plt.plot(data1.train[i][0][0], data1.train[i][0][1], marker='.', color='r')
    plt.plot(data2.train[i][0][0], data2.train[i][0][1], marker='.', color='b')
    plt.plot(data3.train[i][0][0], data3.train[i][0][1], marker='.', color='g')


knn = KNN(traindata, [1, 2, 3])

testdata = np.vstack((data1.test, data2.test, data3.test))
collect = 0
t0 = time.perf_counter()
for i in range(len(testdata)):
    cls = knn.getClass(testdata[i][0], k)
    if cls[0] == testdata[i][1]:
        collect += 1
t1 = time.perf_counter()

collect_rate = collect / len(testdata)
print(collect, len(testdata), collect_rate)
print(t1 - t0)

plt.savefig('./knn.png')
plt.show()
