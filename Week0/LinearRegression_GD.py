import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model


def compute_cost(X, y, theta):
    '''
    Cost Function
    Input:
        X  :  N x 2 matrix
        y  :  N x 1 vector
        theta  : 2 x 1 vector
    Returns
        J  : Cost
    '''
    m = X.shape[0]
    pred = X.dot(theta).flatten()

    sq_error = np.square(pred - y)

    J = sq_error.sum() / 2 / m

    return J





def GradientDescent(X, y, theta, alpha=0.1, iter = 100):
    m = X.shape[0]
    J_history = []

    X_train = np.ones(shape=(m, 2))
    X_train[:, 0] = X

    for i in range(iter):
        pred = X_train.dot(theta).flatten()
        error = (pred -  y)  # (n, )
        # 1 * n  n * 2 = 1 * 2
        theta_diff = error.dot(X_train) / m
        theta -= alpha * theta_diff.reshape(-1, 1)
        J_history.append(compute_cost(X_train, y, theta))

    return theta, J_history

if __name__ == '__main__':
    sample = pd.read_csv("AirPassengers.csv")

    X = sample.X.values

    X = (X - min(X)) / (max(X) - min(X))
    Y = sample.Y.values
    Y = (Y - min(Y)) / (max(Y) - min(Y))
    # 
    #X = np.array([1,2,3,4,5,6,7,8,9,10])
    #Y = 2*X + 1
    lm = linear_model.LinearRegression()
    lm.fit(X.reshape(-1, 1), Y)
    a = lm.coef_
    b = lm.intercept_

    print("Answer: Y = {} X + {}".format(a[0], b))

    theta, J_history = GradientDescent(X, Y, np.ones(shape=(2, 1)))
    print(theta)
    plt.plot(np.arange(len(J_history)) + 1, J_history)
    plt.show()



