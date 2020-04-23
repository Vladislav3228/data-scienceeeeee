import numpy as np
import pandas
from math import sqrt
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import random

def dist(a, b):
    D = 0
    for i in range(len(a)):
        D += (a[i] - b[i]) ** 2
    return sqrt(D)
    
def kernel(center, point, width_point):
  if (center == width_point).any():
    return 1
  else:
    return 1 - (dist(center, point)/dist(center, width_point))**2

#возвращает метку класса для выбранной точки по k ближайшим соседям
def predict(k, center, iris):
  sz = iris.data.shape[0]
  distances = np.zeros([sz, 2])
  for i in range(sz):
    distances[i][0] = dist(center, iris.data[i])
    distances[i][1] = i
  weights = dict()
  sorted_distances = np.sort(distances.view('f8,f8'), order=['f0'], axis=0).view(np.float64)
  for i in range(1, k+1):
    a = int(sorted_distances[i][1])
    b = int(sorted_distances[k+1][1])
    target = iris.target[a]
    mean = float(kernel(center, iris.data[a], iris.data[b]))
    if target in weights:
      weights[target] += mean
    else:
      weights[target] = mean
  sorted_targets = sorted(weights.items(), key=lambda kv: kv[1])
  return sorted_targets[-1][0]

def loo(k, iris):
  errors = 0
  for i in range(iris.data.shape[0]):
    if iris.target[i] != predict(k, iris.data[i], iris):
      errors += 1
  return errors

if __name__ == '__main__':
  iris = load_iris()
  sz = iris.data.shape[0]
  errors = np.zeros(sz - 2)
  count_k = np.zeros(sz - 2)
  for k in range(1, sz - 1):
    errors[k - 1] = loo(k, iris)
    count_k[k - 1] = k
  fig, ax = plt.subplots()
  ax.plot(count_k, errors)
  ax.grid()
  ax.set_xlabel('кол-во соседей')
  ax.set_ylabel('кол-во ошибок')
  plt.show()
