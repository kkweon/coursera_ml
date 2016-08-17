from computeCostMulti import computeCostMulti
from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescent
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data2.txt', delimiter=',')


X = data[:, :-1]
y = data[:, -1]


del data
answers = {}

X, mu, sd = featureNormalize(X)
answers['X_norm'] = X
answers['mu'] = mu
answers['sd'] = sd

y, _, _ = featureNormalize(y)

X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
y = y.reshape(-1, 1)
theta = np.zeros(X.shape[1]).reshape(-1, 1)


theta, J_history = gradientDescent(X, y, theta, alpha=0.1)
print(theta)
print(J_history)


answers['theta'] = theta
answers['J_history'] = J_history

pk.dump(answers, open('ans.p', 'wb'))


