import numpy as np

def getscaleddata():
    X_train = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    X_test = np.array([[1, 0, 0.03], [1, 0, 0.0], [1, 1, 0 ], [1, 1, 0.2]])
    Y_train = np.array([[1], [0], [0], [1]])
    Y_test = np.array([[1.], [1.], [0.], [0.]])
    return X_train, Y_train, X_test, Y_test

def sigmoid(x):
    y = 1 + np.exp(-x)
    y = 1 / y
    return y

def forward(X_training, wt1, wt2):
    temp = np.matmul(X_training, wt1)
    zee1 = sigmoid(temp)
    zee1 = np.concatenate((np.ones((len(zee1), 1)), zee1), axis=1)
    zee2 = sigmoid(np.matmul(zee1, wt2))
    return temp, zee1, zee2

def predict(X_test, w1, w2):
    temp, zee1, zee2 = forward(X_test, w1, w2)
    return zee2

def propagate(X, zee1, zee2, y, w2, temp):
    dif2 = zee2 - y
    D2 = np.matmul(zee1.T, dif2)
    D1 = np.matmul(X.T, (dif2.dot( w2[1:, :].T )) * sigmoid(temp) * (1 - sigmoid(temp)))
    return D1, D2

def updateWeights(D1, D2, alpha, m, w1, w2):
    dw1 = alpha * (1 / m) * D1
    w1 = w1 - dw1
    change_in_w2 = alpha * (1 / m) * D2
    w2 = w2 - change_in_w2
    return w1, w2

def test(X_test, y_test, w1, w2):
    outp = predict(X_test, w1, w2)
    print("\nModel Output: ")
    print((outp))
    print("\nActual Answer")
    print(y_test)

def getweights(X, y, alpha = 0.05, epochs = 10000):
    w1, w2 = np.random.randn(3,5), np.random.randn(6,1)
    for i in range(epochs):
        temp, zee1, zee2 = forward(X, w1, w2)
        D1, D2 = propagate(X, zee1, zee2, y, w2, temp)
        w1, w2 = updateWeights(D1, D2, alpha, len(X), w1, w2)
    return w1, w2

X_train, Y_train, X_test, Y_test = getscaleddata()
w1, w2 = getweights(X_train, Y_train, 0.5, 1000)
test(X_test, Y_test, w1, w2)