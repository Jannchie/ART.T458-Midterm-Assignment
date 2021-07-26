# %% requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv


def grad(w):
  return 2 * np.dot(A, w-mu)


def soft_thresh(y, t):
  res = np.zeros(y.shape)
  for i in range(len(y)):
    if np.linalg.norm(y[i]) > t:
      res[i] = y[i] - t / np.linalg.norm(y[i]) * y[i]
    else:
      res[i] = 0
  return res


# condition
A = np.array([[3,   0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])


def pg(count, lamb=2, lr=0.02):
  w_hat = None
  if lamb == 2:
    w_hat = np.array([0.82, 1.09])
  elif lamb == 4:
    w_hat = np.array([0.64, 0.18])
  elif lamb == 6:
    w_hat = np.array([0.33, 0.0])
  w = np.random.rand(2, 1)
  his = []
  for _ in range(count):
    g = grad(w)
    w = soft_thresh(w - lr * g, lr * lamb)
    his.append(np.linalg.norm(w - w_hat))

  return w, his


# lams = np.arange(0, 6, 0.1)
# w_hat_lam = []
# for lam in lams:
# w_hat = pg(200, 0.02, lam)
# %%
def draw(pg, lam):
  _, his = pg(200, lam)
  plt.plot(range(len(his)), his, label=f"λ={lam}")


draw(pg, 2)
draw(pg, 4)
draw(pg, 6)

plt.semilogy()
plt.legend()
plt.show()
# %%
