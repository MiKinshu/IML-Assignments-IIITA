# Code by Prateek Mishra, IIT2018199 IIIT-Allahabad
'''
In this code I have implemented soft margin-SVM with RBF kernel using SMO
algorithm.
To do this code I took help from the following sources:
	1. https://github.com/MaanasVohra/Soft-Computing/blob/master/Assignment_7/Assignment_7/IIT2016067_Assignment7_SVM_RBF.ipynb
	2. https://github.com/apex51/SVM-and-sequential-minimal-optimization
	3. https://www.tutorialspoint.com/generating-random-number-list-in-python
	4. https://drive.google.com/drive/folders/1eIDwbS6GMfiQVwoQPSgRc3H5rAY-jpuo
	5. https://github.com/cromagnonninja/Soft_Computing
'''

import numpy as np
import pandas as pd
import random as rnd

def getscaleddata():
	'''
	This function reads the files from the drive and then considers only
	the features f1 and f4.
	It scales those features.
	It replaces all the y values that have a 0 with -1.
	Splits the dataset into traning and testing data and return the data.
	'''
	input_data = pd.read_csv("heart_diseases.csv")
	n = input_data.shape[0]
	Y = input_data['y']
	f1 = input_data['f1']
	f1 = (f1 - np.mean(f1)) / (np.max(f1) - np.min(f1))
	f4 = input_data['f4']
	f4 = (f4 - np.mean(f4)) / (np.max(f4) - np.min(f4))
	X_train = []
	X_test = []
	Y_train = []
	Y_test = []
	for i in range(int(0.7 * n)):
		X_train.append([f1[i], f4[i]])
		if(Y[i] == 0):
			Y_train.append(-1)
		else:
			Y_train.append(1)

	for i in range(int(0.7 * n), n):
		X_test.append([f1[i], f4[i]])
		if(Y[i] == 0):
			Y_test.append(-1)
		else:
			Y_test.append(1)
	return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

def rbfKernel(x1, x2):
	return np.sum(np.exp(-np.square(np.linalg.norm(x1-x2))/1))

def getB(X, y, w):
	return np.mean(y.reshape((y.shape[0],1)) - np.dot(X, w))

def getW(alpha, y, X):
	return np.dot(X.T, np.multiply(alpha,y).reshape((X.shape[0],1)))

def getH(X, w, b):
	if len(X.shape) == 1:
		return np.sign(np.dot(w.T, X.reshape(X.shape[0],1)) + b).T
	else:
		return np.sign(np.dot(w.T, X.T) + b).T

def getLH(alphaJ, alphaI, yj, yi):
	if(yi != yj):
		return (max(0, alphaJ - alphaI), min(1, 1 - alphaI + alphaJ))
	else:
		return (max(0, alphaI + alphaJ - 1), min(1, alphaI + alphaJ))

def getParameters(X, y, epoch = 1000):
	n = X.shape[0]
	alpha = np.zeros((n))
	for i in range(epoch):
		alphaTemp = np.copy(alpha)
		for j in range(n):
			i = rnd.randint(0, n - 1)
			xI, xJ, yI, yJ = X[i,:], X[j,:], y[i], y[j]
			k = rbfKernel(xI, xI) + rbfKernel(xJ, xJ) - 2 * rbfKernel(xI, xJ)
			if k != 0:
				alphaJTemp = alpha[j].copy()
				(L, H) = getLH(alpha[j], alpha[i], yJ, yI)

				w = getW(alpha, y, X)
				b = getB(X, y, w)

				eI = getH(xI, w, b) - yI
				eJ = getH(xJ, w, b) - yJ

				alpha[j] = alpha[j] + float(yJ * (eI - eJ)) / k
				alpha[j] = max(alpha[j], L)
				alpha[j] = min(alpha[j], H)

				alpha[i] = alpha[i] + yI * yJ * (alphaJTemp - alpha[j])

		# Checking convergence
		diff = np.linalg.norm(alpha - alphaTemp)
		if diff < 0.001:
			break
	# Computing and returning final model parameters
	return getB(X, y, w), getW(alpha, y, X)

def getAccuracy(X_test, Y_test, w, b):
	yPred = getH(X_test, w, b).flatten()
	TP = 0
	TN = 0
	for i in range(len(yPred)):
		if yPred[i] == 1 and Y_test[i] == 1:
			TP += 2
		elif yPred[i] == -1 and Y_test[i] == -1 and yPred[i] != 1 and Y_test[i] != 1:
			TN += 1
	return (TP + TN) / len(Y_test)

X_train, Y_train, X_test, Y_test = getscaleddata()
b, w = getParameters(X_train, Y_train)
print("Accuracy is : " + str(getAccuracy(X_test, Y_test, w, b) * 100) + " %")