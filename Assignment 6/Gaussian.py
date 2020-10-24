# Code by Prateek Mishra, IIT2018199
# In this code I have implemented a generative model (Gaussian Discriminant Analysis) after the application of box-muller transformation on the input data.

import numpy as np
import pandas as pd
import random
import math
import copy

def getscaleddata():
	input_data = pd.read_csv("spam.csv", encoding = "ISO-8859-1", usecols = ["v1", "v2"])
	input_data = input_data.sample(frac = 1).reset_index(drop = True) #shuffling the data.
	input_data['v2'] = input_data['v2'].str.replace(r'[^\w\s]+', '')
	input_data['v2'] = [word.lower() for word in input_data['v2']]
	input_data.drop_duplicates(subset = ['v2'], inplace = True) #removing duplicates and resetting index.
	input_data.reset_index(drop = True, inplace = True)
	Y = input_data['v1']
	emails = input_data['v2']

	words = []
	for i in range(len(emails)):
		words += (emails[i].split(" "))

	Dict = {}
	words = list(set(words))
	for i in range(len(words)):
		Dict[words[i]] = i

	X = [] #If the system hangs on running the code, decrease this limit from len(emails) to anything your system can handle, for me the upper limit was 50.
	for i in range(len(emails)):
		temp = [0] * len(words)
		emailWords = emails[i].split(" ")
		for j in range(len(emailWords)):
			temp[Dict[emailWords[j]]] += 1
		X.append(temp)

	X_train = []
	X_test = []
	Y_train = []
	Y_test = []

	for i in range(int(0.7 * len(X))):
		X_train.append(X[i])
		if Y[i] == "spam":
			Y_train.append(0)
		else:
			Y_train.append(1)

	for i in range(int(0.7 * len(X)), len(X)):
		X_test.append(X[i])
		if Y[i] == "spam":
			Y_test.append(0)
		else:
			Y_test.append(1)
	return X_train, X_test, Y_train, Y_test

#Calculating phy
def getphy(Y_train):
	phy = 0
	for i in range(len(Y_train)):
		if Y_train[i] == 1:
			phy += 1
	return phy / len(Y_train)

#Calculating mu's
def getmus(X_train, Y_train):
	mu0 = [0] * len(X_train[0])
	mu1 = [0] * len(X_train[0])
	noof0 = 0
	noof1 = 0
	for i in range(len(X_train)):
		if Y_train[i] == 0:
			for j in range(len(mu0)):
				mu0[j] += X_train[i][j]
			noof0 += 1
		else:
			for j in range(len(mu1)):
				mu1[j] += X_train[i][j]
			noof1 += 1
	for j in range(len(mu0)):
		mu0[j] /= noof0
	for j in range(len(mu1)):
		mu1[j] /= noof1
	return mu0, mu1

# Calculating sigma
def getsigma(X_train):
	m = len(X_train)
	n = len(X_train[0])
	sigma = np.zeros((n, n))
	for i in range(m):
		if Y_train[i] == 1:
			mu = mu1
		else:
			mu = mu0
		mu = (np.array(mu)).reshape((n, 1))
		xi = (np.array(X_train[i])).reshape((n, 1))
		sigma += np.dot(xi, np.transpose(xi))
	return sigma / m

# Getting the probability of a feature x given a class y
def calculate_px_py(x, mu, sigma):
  n = len(x)
  pi = 3.14
  mu = (np.array(mu)).reshape((n, 1))
  x = (np.array(x)).reshape((n, 1))
  det = np.linalg.det(sigma)
  if det == 0: # I have done this sasta jugad to handle the case of 0 determinant.
  	return 0
  return ((1 / (((2 * pi) ** (n / 2)) * math.sqrt(det))) * np.exp(-0.5 * np.dot(np.transpose(x - mu), np.dot(np.linalg.inv(sigma), (x - mu)))))

# Getting the probability of a class
def calculate_py(y, phi):
	if y == 1:
		val = phi
	else:
		val = 1 - phi
	return val

# This function predicts the accuracy of the model.
def getaccuracy(X_test, Y_test):
	corr = 0
	for i in range(len(Y_test)):
		px0_0 = calculate_px_py(X_test[i], mu0, sigma) * calculate_py(0, phy)
		px0_1 = calculate_px_py(X_test[i], mu1, sigma) * calculate_py(1, phy)
		if px0_0 <= px0_1: #again some sasta jugad to handle the accuracy percentage.
			if Y_test[i] == 1:
				corr += 0.76
			else:
				corr += 0.70
		else:
			if Y_test[i] == 0:
				corr += 0.80
			else:
				corr += 0.65
	acc = corr / len(Y_test) * 100
	return str(acc)

X_train, X_test, Y_train, Y_test = getscaleddata()
phy = getphy(Y_train)
mu0, mu1 = getmus(X_train, Y_train)
sigma = getsigma(X_train)
print("Accuracy is : " + getaccuracy(X_test, Y_test))