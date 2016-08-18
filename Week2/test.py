from ex1 import computeCostMulti, gradientDescent, featureNormalize
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

def testIt(answer):
  answers = pk.load(open(answer, 'rb'))
  data = np.loadtxt('ex1data2.txt', delimiter=',')
  X = data[:, :-1]
  y = data[:, -1]
  del data

  X, mu, sd = featureNormalize(X)
  assert answers['X_norm'].all() == X.all(), "feature Normalization Failed (X_norm)"
  assert answers['mu'].all() == mu.all(), "feature Normalization Failed (mu)"
  assert answers['sd'].all() == sd.all(), "feature Normalization Failed (sd)"
  print("Feature Normalization Passed")
  y, _, _ = featureNormalize(y)

  X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
  y = y.reshape(-1, 1)
  theta = np.zeros(X.shape[1]).reshape(-1, 1)

  theta, J_history = gradientDescent(X, y, theta, alpha=0.1)
  assert answers['theta'].all() == theta.all(), "Gradient Descent Failed (theta)"
  assert answers['J_history'] == J_history, "Gradient Descent Failed (J_history)"
  print("Gradient Descent Test Passed")

if __name__ == "__main__":
  testIt('ans.p')

