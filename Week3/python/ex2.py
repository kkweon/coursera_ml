import numpy as np

def sigmoid(X):
  pass
  return g

def costFunction(theta, X, y):
  pass
  return J, grad

def predict(theta, X):
  pass
  return p

def mapFeature(X1, X2):
  degree = 6
  X = np.ones(X1.shape[0]).reshape(-1, 1)
  for i in range(1, degree+1):
    for j in range(i+1):
      toAppend = X1**(i-j) * X2**j
      X = np.concatenate((X, toAppend), axis=1)
  return X

def costFunctionReg(theta, X, y, lmda):
  pass
  return J, grad
