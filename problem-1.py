#%%
import numpy as np
from dataset import ds_4, ds_5
from matplotlib import pyplot as plt


def J(w):
  reg = lamb * np.dot(w.T, w)
  cost = 0
  for i in range(x.shape[0]):
    cost += np.log(1 + np.exp(-y[i] * np.dot(w.T, x[i])))
  return reg + cost


def grad(w):
  res = 2 * lamb * w
  for i in range(x.shape[0]):
    res += (-y[i] * x[i] * (1 - 1 / (1 + np.exp(-y[i] * np.dot(w.T, x[i])))))
  return res


def bgd():
  w = np.random.rand(4)
  loss_hist_batch = []
  for _ in range(100):
    loss = J(w)
    g = grad(w)
    w -= g * lr
    loss_hist_batch.append(loss)
  return loss_hist_batch


def get_h(w):
  res = 2 * lamb * np.eye(4)
  for i in range(x.shape[0]):
    e = np.exp(-y[i] * np.dot(w.T, x[i]))
    res += y[i]**2 * e / (1 + e)**2 * np.dot(x[i], x[i].T)
  return res


def newton():
  w = np.random.rand(4)
  loss_list = []
  for _ in range(100):
    loss = J(w)
    g = grad(w)
    h = get_h(w)
    w -= lr * np.dot(np.linalg.inv(h), g)
    loss_list.append(loss)
  return loss_list


#%%


def problem_1_2(x, y):
  loss_hist_batch = bgd()
  loss_newton_batch = newton()
  best = loss_newton_batch[-1] if loss_newton_batch[-1] < loss_hist_batch[
      -1] else loss_hist_batch[-1]
  lh = loss_hist_batch - best
  ln = loss_newton_batch - best
  plt.plot(lh, label="BGD")
  plt.plot(ln, label="Newton")
  plt.semilogy()
  plt.legend()
  plt.show()


# %%
lamb = 0.05
lr = 0.01

# %%
x, y = ds_4()
problem_1_2(x, y)
# %%
lamb = 1
x, y = ds_5()
problem_1_2(x, y)
# %%
