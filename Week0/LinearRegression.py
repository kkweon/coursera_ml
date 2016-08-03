import pandas as pd 
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
'''
(0,0)
(1,1)
(2,2)

y = a x + b where a = 1 and b = 0
'''

FIG_FOLDERS = "./fig/"
try:
    for file in os.listdir(FIG_FOLDERS):
        os.remove(os.path.join(FIG_FOLDERS, file))
except:
    print("Couldn't delete files")


def plot_fig(X, Y, Y_pred, coefficients_results, i, a, b):
    plt.figure(1)
    plt.subplot(121)
    plt.plot(np.array(coefficients_results)[:,0], np.array(coefficients_results)[:,1], 'bo--', alpha=0.1)
    plt.xlabel("Slope")
    plt.ylabel("Y-Intercept")

    plt.subplot(122)            
    plt.plot(X[:, 0], Y.flatten(), 'ro', label='Sample')
    plt.plot(X[:, 0], Y_pred.flatten(), 'b-', label='Y={:.1f}X+{:.1f}'.format(a,b))
    plt.xlim(min(X[:,0]), max(X[:, 0]))
    plt.ylim(min(Y.flatten()), max(Y.flatten()))
    plt.legend()
    plt.title("{} Iteration Done".format(i))
    plt.savefig("fig/{}.png".format(i))
    plt.close()

def gradient_descent(X, Y, learning_rate = 0.1, max_iteration = 30000):
    ## y = ax + b
    ## y
    a = 0
    b = 0

    X = np.array(X)
    temp_X = np.ones((X.shape[0], 2))
    temp_X[:, 0] = X
    X = temp_X
    Y = np.array(Y).flatten()
    coefficient_results = [(a, b)]
    coefficient = np.array([a, b])

    error_lists = []
    for i in range(max_iteration):
        Y_pred = np.matmul(X, coefficient)
        MISS_A = Y_pred - Y
        ERROR = (np.sum(np.square(MISS_A))) / 2 / X.shape[0]

        error_lists.append(ERROR)

        if i % 10 == 0:
            print("Y = {} X + {} (Iteration: {}, ERROR: {})".format(a, b, i, ERROR))
            plot_fig(X, Y, Y_pred, coefficient_results, i, a, b)

        if ERROR <= 0.001:
            plot_fig(X, Y, Y_pred, coefficient_results, i, a, b)
            return a, b, i, ERROR

        else:
            if len(error_lists) > 2:
                if error_lists[-2] == ERROR:
                    plot_fig(X, Y, Y_pred, coefficient_results, i, a, b)
                    return a, b, i, ERROR
            delta = np.matmul((MISS_A).reshape(1,-1), X) / X.shape[0]
            coefficient  = coefficient - learning_rate * delta.flatten()            
            a = coefficient[0]
            b = coefficient[1]
            if math.isnan(a) or math.isnan(b):
                return a, b, i, ERROR
            elif math.isinf(a) or math.isinf(b):
                print("Inf Found!") 
                print(np.mean(Y_pred-Y))
                return a, b, i, ERROR

            coefficient_results.append((a,b))

    plot_fig(X, Y, Y_pred, coefficient_results, i, a, b)
    return a, b, max_iteration, ERROR

if __name__ == "__main__":
    #X = np.random.rand(50) * np.random.rand() + 10
    #Y = np.random.rand()+10 * X + np.random.rand() + 10
    #X = [12.39999962, 14.30000019, 14.5, 14.89999962, 16.10000038, 16.89999962, 16.5, 15.39999962, 17, 17.89999962, 18.79999924, 20.29999924, 22.39999962, 19.39999962, 15.5, 16.70000076, 17.29999924, 18.39999962, 19.20000076, 17.39999962, 19.5, 19.70000076, 21.20000076]
    #Y = [11.19999981, 12.5, 12.69999981, 13.10000038, 14.10000038, 14.80000019, 14.39999962, 13.39999962, 14.89999962, 15.60000038, 16.39999962, 17.70000076, 19.60000038, 16.89999962, 14, 14.60000038, 15.10000038, 16.10000038, 16.79999924, 15.19999981, 17, 17.20000076, 18.60000038]


    SampleData = pd.read_csv("AirPassengers.csv")
    X = SampleData.X.reshape(-1, 1)
    Y = SampleData.Y.reshape(-1, 1)

    X_max = max(X)
    X_min = min(X)
    Y_max = max(Y)
    Y_min = min(Y)

    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    regr = linear_model.LinearRegression()
    regr.fit(X, Y) 
    print("Linear Regression with Scikit-learn Coefficients: Y = {} X + {}".format(regr.coef_.flatten()[0], regr.intercept_[0]))


    X = (X - X_min) / X_range
    Y = (Y - Y_min) / Y_range

    a_old, b, iteration, ERROR = gradient_descent(X.flatten(), Y.flatten())

    a = a_old * Y_range / X_range 
    b = Y_range * (b - a_old * X_min / X_range) + Y_min
    print("{} iterations Done\n Y = {}X + {} with Error of {}".format(iteration, a, b, ERROR))