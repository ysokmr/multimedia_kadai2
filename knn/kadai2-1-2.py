import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from mkpicdata import make_image_set

import time

k = 3
train_size = 10000
test_size = 1000


traindata = make_image_set(train_size)

knn = KNN(traindata, [0, 1, 2, 3])

testdata = make_image_set(test_size)
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
