#computeCostMulti(X, y, theta)
#
import numpy as np

def computeCostMulti(X, y, theta):
  '''
  Compute Cost for Multi Variables Linear Regressions

  Input:
  =============
  Y = theta0 * x0 + theta1 * x1 + ... + theta_n * x_n
  X  :  Variables (numpy array of N X (D + 1))
  y  :  Y Predicted Value (numpy array of N x 1)
  theta  :  Parameters (numpy array of (N+1,)

  Output
  ======================
  J  :  cost (float)
  '''

  y_pred = X.dot(theta)

  cost = (y_pred - y)**2 # y diff (y - y_pred)

  J = np.mean(cost)

  return J


