# Code by Prateek Mishra, IIT2018199
# In this code I have implemented Implement Perceptron training algorithms for AND, OR, NAND and NOR gates. To choose different gates simply comment out the ones required ones from the getscaled data function.

import numpy as np

def getscaleddata():
    X_train = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    X_test = [[0.98, 1],[0.01, 0.97],[0.77, 0.99],[0.912, 1.002],[0.88, 0.11],[0.82, 0.9],[0.8, 1],[0.02, 0.01],[0.21, 0.99],[0.11, 0.2],[0.79, 1],[0.11, 1.02],[0.98, 0.87],[0.2, 1.3],[0.2, 0.003]]
    Y_train = [1, 1, 1, 0]
    # Y_test = [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0] # AND
    # Y_test = [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0] # OR
    Y_test = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1] # NOR
    # Y_test = [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1] # NAND
    return np.asarray(X_train), np.asarray(Y_train), np.asarray(X_test), np.asarray(Y_test)

def getweights(X_train, Y_train, alpha = 0.35, epochs = 100):
    w0, w1, w2 = np.random.randn(), np.random.randn(), np.random.randn()
    for i in range(epochs):
        for j in range(len(X_train)):
            if (w0 * X_train[j][0] + w1 * X_train[j][1] + w2 * X_train[j][2] >= 0):
                act = 1
            else:
                act = 0
            w0 = w0 + alpha * X_train[j][0] * (Y_train[j] - act)
            w1 = w1 + alpha * X_train[j][1] * (Y_train[j] - act)
            w2 = w2 + alpha * X_train[j][2] * (Y_train[j] - act)
    return w0, w1, w2

def getaccuracy(X_test, Y_test, w0, w1, w2):
    corr = 0
    for i in range(len(X_test)):
        if (w0 + w1 * X_test[i][0] + w2 * X_test[i][1] >= 0):
            ans = 1
        else:
            ans = 0
        if (ans == Y_test[i]):
            corr += 1
    return str(corr / len(Y_test) * 100) + " %"

X_train, Y_train, X_test, Y_test = getscaleddata()
w0, w1, w2 = getweights(X_train, Y_train, 0.30, 200)
print("Accuracy is : " + getaccuracy(X_test, Y_test, w0, w1, w2))