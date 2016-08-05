import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def compute_cost(X, y, theta):
  pred = X.dot(theta).flatten()
  sq_error = np.square(pred - y)
  J = sq_error.sum() / 2.0 / X.shape[0]

  return J

def gradient_descent(X, y, theta=np.ones(shape=(2,1)), learning_rate=0.01, max_iteration=10000000):
  X_train = np.ones(shape=(X.shape[0], 2))
  X_train[:, 0] = X
  J_history = []
  m = X.shape[0]
  for i in range(max_iteration):
    y_pred = X_train.dot(theta).flatten()
    error = (y_pred - y)  
    theta_diff = error.dot(X_train) / m
    theta = theta - learning_rate * theta_diff.reshape(-1, 1)
    cost = compute_cost(X_train, y, theta)
    if len(J_history) > 1 and np.abs(cost - J_history[-1]) < 0.000000000000001:
      J_history.append(cost)
      return theta, J_history
    J_history.append(cost)
    if i%1000 == 0:
      print("{}\tIteration \t=> Y = {} X + {}  with ERROR of {}".format(i, theta[0], theta[1], J_history[-1]))

  return theta, J_history




if __name__ == "__main__":
  dataset = "./ex1data1.txt"
  X, Y = [], []
  with open(dataset, 'r') as f:
    data = f.readlines()
    for line in data:
      line = line.split(",")
      X.append(float(line[0]))
      Y.append(float(line[1]))
  X = np.array(X)
  Y = np.array(Y)

  # X = np.array([1,2,3,4,5, 6, 7, 8, 9, 10])
  # Y = 2*X + 3

  print(len(X)==len(Y))
  print(X.shape, Y.shape)
  regr = linear_model.LinearRegression() 
  regr.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
  print("Y = {} X + {}".format(regr.coef_[0][0], regr.intercept_[0]))
  theta, J_history = gradient_descent(X, Y)
  print("Expected\nY = {} X + {}".format(theta[0], theta[1]))
  plt.plot(range(len(J_history)), J_history)
  plt.show()

