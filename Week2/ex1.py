import numpy as np

def featureNormalize(X):
  """
  Returns Mean Normalized X

  Input
  ============================
  X  :  np.array of (N, D)

  Returns
  ============================
  X_norm : np.array of (N, D)
  mu  :  np.array of (D, 1)
  sd  :  np.array of (D, 1)
  """ 
  X_norm = X
  mu = np.mean(X_norm, axis=0)
  sd = np.std(X_norm, axis=0)
  X_norm = (X_norm - mu) / sd

  return X_norm, mu, sd

def computeCostMulti(X, y, theta):
  '''
  Compute Cost for Multi Variables Linear Regressions

  Y = theta_0 * x_0 + theta_1 * x_1 + ... + theta_D * x_D

  Input:
  =============
  X  :  numpy array of (N, D+1)
  
  ex)  X = 

  1    x11    x12    ...    x1D
  1    x21    x22    ...    x2D
  1    x31    x32    ...    x3D
  .    ...    ...    ...    ...
  1    xN1    xN2    ...    xND


  y  :  numpy array of (N, 1) #주의 (N, ) 아님 

  ex)  y =
  
  y0
  y1
  ...
  yN
  
  theta  :  numpy array of (D+1, 1)
 
  ex)  theta = 

  theta_0
  theta_1
  ...
  theta_D


  Output
  ======================
  J  :  cost (float)
  '''
  y_pred = X.dot(theta)
  cost = (y_pred - y)**2 # y diff (y - y_pred)
  J = np.mean(cost)

  return J

def gradientDescent(X, y, theta, alpha = 0.01, num_iters = 1500):
  '''
  Calculate Gradient Descent

  Input
  ===================================
  X         : np.array (N, D+1)
  y         : np.array (N, 1)
  theta     : np.array (D+1, 1)
  alpha     : learning rate (float)
  num_iters : max iterations (int)

  Returns
  ===================================
  theta       :  np.array (D+1, 1)
  J_history   :  list containing cost J of each iterations
  '''
  m = X.shape[0]
  y = y.reshape(-1, 1)
  theta = theta.reshape(-1, 1)
  J_history = []

  for _ in range(num_iters):
    y_pred = X.dot(theta)
    error = y_pred - y
    dTheta = X.T.dot(error) / m
    theta = theta - alpha * dTheta
    J_history.append(computeCostMulti(X, y, theta))

  return theta, J_history


