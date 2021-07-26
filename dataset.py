import numpy as np


def ds_4():
  '''dataset IV'''
  n = 200
  x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
  y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:, 1] + 0.5 + 0.5 * np.random.randn(n)) > 0
  y_d4 = 2 * y_d4 - 1
  return x_d4, y_d4
