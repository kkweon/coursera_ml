'''
BASE FILE for Exercise 3
==========================
Logistics Regressions for Multiple Classes
Neural Networks
'''
import numpy as np
import scipy.io as sio
from scipy.optimize import fmin_cg, fmin_bfgs, minimize

def sigmoid(z):
  '''Returns Sigmoid(X)

  :param z: any value
  :returns: float of the given shape

  '''
  return 1 / (1 + np.exp(-z))

def reshape(x):
  '''Simple Helper Function that turns np.array(n, ) to np.array(n, 1)

  :param x: np.array(N, )
  :returns: np.array(N, 1)

  '''
  return x.reshape(-1, 1)

'''
Assignment Begins
-----------------------------

'''

def lr_cost_function(theta, X, y, lamda):
  """Returns :math:`Cost(J) = -y * log(sigmoid(X * \\theta)) - (1 - y) * log(1 - sigmoid(X*\\theta))`

  :param theta: np.array(D, 1)
  :param X: np.array(N, D)
  :param y: np.array(N, 1)
  :param lamda: float -- hyper parameter for regularization strength.
  :returns: float -- cost J

  """
  y = reshape(y)
  theta = reshape(theta)

  m = X.shape[0]

  z = X.dot(theta)
  h = sigmoid(z)
  J_i = -y * np.log(h) - (1 - y) * np.log(1 - h)
  J = np.sum(J_i) / m + np.sum(np.square(theta[1:, :])) * lamda / (2 * m)

  return J

def lr_grad_function(theta, X, y, lamda):
  '''Calculate Grad

  .. todo::
  
    잘 동작하고 있는 것 같지 않다.

  :param theta: np.array(D, 1)
  :param X: np.array(N, D)
  :param y: np.array(N, 1)
  :param lamda: float -- hyper parameter for regularization strength.
  :returns: np.array(D, )

  '''
  y = reshape(y)

  theta = reshape(theta)

  m = X.shape[0]
  z = X.dot(theta)
  h = sigmoid(z)

  temp = theta
  temp[0] = 0

  err = (h - y).reshape(-1, 1)
  grad = X.T.dot(err) / m + temp.reshape(-1, 1) * lamda / m
  
  return grad.flatten()

def oneVsAll(X, y, num_labels, lamda, verbose=1):
  '''Calculate Theta for multi classes :math: `X_0`
 
  :param X: np.array(N, D)
  :param y: np.array(N, 1)
  :param num_labels: int -- number of labels
  :param lamda: float -- for regularization strength

  :returns: np.array(C, D) -- **all_theta**

  '''
  X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)
  all_theta = np.zeros(shape=(num_labels, X.shape[1]))

  for i in range(num_labels):
    initial_theta = np.random.randn(X.shape[1], 1) + 1e-9
    #theta_found = fmin_cg(lr_cost_function, initial_theta, fprime=lr_grad_function, args=(X, (y==i+1), lamda), disp=verbose)
    theta_found = fmin_bfgs(lr_cost_function, initial_theta, fprime=lr_grad_function, args=(X, (y==i+1), lamda), disp=verbose)
    all_theta[i, :] = theta_found.flatten()
    # theta_found = minimize(lr_cost_function, initial_theta, jac=lr_grad_function, args=(X, (y==i+1), lamda)) 
    # print(theta_found.success)
    # all_theta[i, :] = theta_found.x.flatten()
  return all_theta

def predictOneVsAll(all_theta, X):
  ''' Predict y given X

  .. note::

        The bias term(`X_0`) must be added 

  :param all_theta: C classes D dimensions
  :type all_theta: np.array(C, D)
  :param X: Input Matrix. Bias term should be added before.
  :type X: np.array(N, D)
  :returns: np.array(N,1) -- correct classes
  '''
  X = np.append(np.ones(shape=(X.shape[0], 1)), X, axis=1)
  z = (X).dot(all_theta.T)
  h = sigmoid(z)

  max_indices = np.argmax(h, axis=1)
  p = max_indices + 1
  return p.reshape(-1, 1)

def predict(Theta1, Theta2, X):
  '''Do the forward propagation of given Theta1, Theta2, and X.

  :param Theta1: np.array (given)
  :param Theta2: np.array (given)
  :param X: input (given)
  :returns: np.array(N,1) -- correct classes

  >>> predict(Theta1, Theta2, X)
  np.array([1,5,2,...])
  '''
  pass
  return p

if __name__ == '__main__':
  data1 = sio.loadmat('ex3data1.mat')
  data2 = sio.loadmat('ex3weights.mat')

  X = np.array(data1['X'])
  y = np.array(data1['y'])

  Theta1 = data2['Theta1']
  Theta2 = data2['Theta2']

  del data1, data2

  num_labels = 10
  l = 0.1

  all_theta = oneVsAll(X, y, num_labels, l, verbose=0)
  p = predictOneVsAll(all_theta, X)

  print("Accuracy: {:.2%} <--- should be greater than 94.9%".format(np.mean(p==y)))

 