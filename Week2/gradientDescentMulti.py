from computeCostMulti import computeCostMulti
import numpy as np

def gradientDescent(X, y, theta, alpha = 0.01, num_iters = 1500):
  '''
  Calculate Gradient Descent

  Input
  ========================
  X : np.array (N, D+1)
  y : np.array (N, )

  theta : np.array (D+1, )
  alpha : learning rate (float)
  
  num_iters : max iterations (int)


  Returns
  ==========================
  theta  :  np.array (D+1, )
  J_history  :  list containing costs of each iterations
  '''
  m = X.shape[0]
  J_history = []

  for _ in range(num_iters):
    y_pred = X.dot(theta)
    error = y_pred - y
    dTheta = X.T.dot(error) / m
    theta = theta - alpha * dTheta
    J_history.append(computeCostMulti(X, y, theta))

  return theta, J_history




