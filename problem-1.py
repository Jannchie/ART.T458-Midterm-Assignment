import numpy as np
from dataset import ds_4
from matplotlib import pyplot as plt

x, y = ds_4()
lamb = 0.3


def J(w):
  reg = lamb * np.dot(w.T, w)
  cost = 0
  for i in range(x.shape[0]):
    cost += np.log(1 + np.exp(-y[i] * np.dot(w.T, x[i])))
  return reg + cost


def grad(w):
  res = 2 * lamb * w
  for i in range(x.shape[0]):
    res += (-y[i] * x[i] *
            (1 - 1 / (1 + np.exp(-y[i] * np.dot(w.T, x[i]))))).reshape(4, 1)
  return res


def bgd():
  lr = 0.02
  w = np.random.rand(4, 1)
  loss_hist_batch = []
  for _ in range(30):
    loss = J(w)
    g = grad(w)
    w -= g * lr
    loss_hist_batch.append(loss[0][0])
  return loss_hist_batch


def newton():
  alpha = 0.02
  w = np.random.rand(4, 1)
  loss_list = []
  for _ in range(30):
    loss = J(w)
    g = grad(w)
    h = get_hessian(w)
    w -= alpha * np.dot(np.linalg.inv(h), g)
    loss_list.append(loss[0][0])
  return loss_list


def get_hessian(w):
  hessian = 2 * lamb * np.eye(4)
  for i in range(x.shape[0]):
    e = np.exp(-y[i] * np.dot(w.T, x[i]))
    hessian += y[i]**2 * e / (1 + e)**2 * np.dot(x[i], x[i].T)
  return hessian


loss_hist_batch = bgd()
loss_newton_batch = newton()

plt.plot(loss_hist_batch)
plt.plot(loss_newton_batch)
plt.semilogy()
plt.show()
