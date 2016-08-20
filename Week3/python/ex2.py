import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

def featureNormalize(X):
  X_norm = X
  mu = np.mean(X_norm, axis=0)
  sd = np.std(X_norm, axis=0)
  X_norm = (X_norm - mu) / sd

  return X_norm, mu, sd

def sigmoid(X):
  """Summary
  Returns a sigmoid X
  
  Args:
      X (np.array): Input size of (Number of samples, Dimensions)
  
  Returns:
     g (np.array): returns sigmoid(X)
  """
  g = 1/(1 + np.exp(-X))
  return g

def costFunction(theta, X, y):
  theta = theta.reshape(-1, 1)
  Z = X.dot(theta)
  s = sigmoid(Z)
  J_i = -y * np.log(s)  - (1 - y) * np.log(1 - s)
  J = np.mean(J_i) 

  return J

def gradientDescent(theta, X, y):
  theta = theta.reshape(-1, 1)
  Z = X.dot(theta)
  s = sigmoid(Z)
  error = y - s
  grad = X.T.dot(error) / - X.shape[0]
  return grad.flatten()

def predict(theta, X):
  theta = theta.reshape(-1, 1)
  Z = X.dot(theta)
  s = sigmoid(Z)
  s[s >= 0.5] = 1
  s[s < 0.5] = 0

  return s

def mapFeature(X1, X2):
  degree = 6
  X = np.ones(X1.shape[0]).reshape(-1, 1)
  for i in range(1, degree+1):
    for j in range(i+1):
      toAppend = X1**(i-j) * X2**j
      X = np.append(X, toAppend, axis=1)
  return X

def costFunctionReg(theta, X, y, l):
  m = X.shape[0]
  J = costFunction(theta, X, y)
  J += np.sum(theta[1:]**2) * l / (2 * m)
  return J

def gradientDescentReg(theta, X, y, l):
  m = X.shape[0]
  grad = gradientDescent(theta, X, y)
  theta[0] = 0
  grad += theta.flatten() * l / m 
  return grad


if __name__=="__main__":
  data1 = np.loadtxt('data/ex2data1.txt', delimiter=",")
  X = data1[:, :-1]
  y = data1[:, -1].reshape(-1, 1)
  X, _, _ = featureNormalize(X)
  X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)

  # Gradient Decsent
  iter = 1600
  lmda = 1
  initial_theta = np.random.randn(X.shape[1]) * 0.1
  J_history = []
  for _ in range(iter):
    J = costFunction(initial_theta, X, y)
    grad = gradientDescent(initial_theta, X, y)
    J_history.append(J)
    initial_theta -= lmda * grad

  y_pred = predict(initial_theta, X)
  RESULT = """
  ===================================
  EXAMPLE 1) expected 89% or greater
  Full batch Gradient Descent

  Initial Loss:    {}
  Train Accuracy:  {:.2%}
  Final Loss:      {}    
  ===================================
  """.format(J_history[0], np.mean(y_pred==y), J_history[-1])
  print(RESULT)

  # Using scikit opt function
  iter = 1600
  lmda = 1
  initial_theta = np.random.randn(X.shape[1]).reshape(-1, 1) * 0.1
  result = fmin_bfgs(costFunction, initial_theta, fprime=gradientDescent, args = (X, y), maxiter=iter, disp=0, full_output=1)
  best_theta = result[0]
  final_loss = result[1]
  y_pred = predict(best_theta, X)
  tr_acc = np.mean(y_pred==y)
  RESULT = """
  ===================================
  EXAMPLE 2) expected 83% or greater
  Scikit optimization (fmin_bfgs)

  Initial Loss:    {}
  Train Accuracy:  {:.2%}
  Final Loss:      {}    
  ===================================
  """.format("Unknown", tr_acc, final_loss)
  print(RESULT)

  data2 = np.loadtxt('data/ex2data2.txt', delimiter=',')
  X = data2[:, :-1]
  y = data2[:, -1].reshape(-1, 1)

  X, _, _ = featureNormalize(X)

  X = mapFeature(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))

  iter = 1600
  l = 1
  lmda = 1.5
  initial_theta = np.random.randn(X.shape[1]) * 0.1
  J_history = []
  for _ in range(iter):
    J = costFunctionReg(initial_theta, X, y, l)
    grad = gradientDescentReg(initial_theta, X, y, l)
    J_history.append(J)
    initial_theta -= lmda * grad
    lmda *= 0.95

  y_pred = predict(initial_theta, X)
  RESULT = """
  ===================================
  EXAMPLE 2) expected 83% or greater
  Full batch Gradient Descent

  Initial Loss:    {}
  Train Accuracy:  {:.2%}
  Final Loss:      {}    
  ===================================
  """.format(J_history[0], np.mean(y_pred==y), J_history[-1])
  print(RESULT)

  # Using scikit opt function
  iter = 5000
  lmda = 1
  initial_theta = np.random.randn(X.shape[1]).reshape(-1, 1) * 0.1
  result = fmin_bfgs(costFunction, initial_theta, fprime=gradientDescent, args = (X, y), maxiter=iter, disp=0, full_output=1)
  best_theta = result[0]
  final_loss = result[1]
  y_pred = predict(best_theta, X)

  tr_acc = np.mean(y_pred==y)
  RESULT = """
  ===================================
  EXAMPLE 2) expected 83% or greater
  Scikit Optimization (fmin_bfgs) 

  Initial Loss:    {}
  Train Accuracy:  {:.2%}
  Final Loss:      {}    
  ===================================
  """.format("Unknown", tr_acc, final_loss)
  print(RESULT)
