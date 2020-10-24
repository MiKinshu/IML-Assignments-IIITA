# Code by Prateek Mishra, IIT2018199
# In this code I have implemented a generative model (Gaussian Discriminant Analysis) on the input data.

import numpy as np
import pandas as pd
import random
import math

def getscaleddata():
	input_data = pd.read_csv("microchip.csv")
	input_data = input_data.sample(frac = 1).reset_index(drop=True) #shuffling the data.
	Y = input_data['y']
	marks1 = input_data['marks1']
	marks2 = input_data['marks2']

	meanmarks1 = np.mean(marks1)
	maxmarks1 = np.max(marks1)
	minmarks1 = np.min(marks1)

	meanmarks2 = np.mean(marks2)
	maxmarks2 = np.max(marks2)
	minmarks2 = np.min(marks2)

	marks1sq = []
	for i in marks1:
		marks1sq.append(i * i)
	meanmarks1sq = np.mean(marks1sq)
	maxmarks1sq = np.max(marks1sq)
	minmarks1sq = np.min(marks1sq)
	for i in range(len(marks1sq)):
		marks1sq[i] = (marks1sq[i] - meanmarks1sq) / (maxmarks1sq - minmarks1sq)

	marks2sq = []
	for i in marks2:
		marks2sq.append(i * i)
	meanmarks2sq = np.mean(marks2sq)
	maxmarks2sq = np.max(marks2sq)
	minmarks2sq = np.min(marks2sq)
	for i in range(len(marks2sq)):
		marks2sq[i] = (marks2sq[i] - meanmarks2sq) / (maxmarks2sq - minmarks2sq)

	marks1marks2 = []
	for i in range(len(marks1)):
		marks1marks2.append(marks1[i] * marks2[i])
	meanmarks1marks2 = np.mean(marks1marks2)
	maxmarks1marks2 = np.max(marks1marks2)
	minmarks1marks2 = np.min(marks1marks2)
	for i in range(len(marks1marks2)):
		marks1marks2[i] = (marks1marks2[i] - meanmarks1marks2) / (maxmarks1marks2 - minmarks1marks2)

	X_train = []
	X_test = []
	Y_train = []
	Y_test = []

	for i in range(int(0.7 * len(input_data))):
		X_train.append([(marks1[i] - meanmarks1) / (maxmarks1 - minmarks1), (marks2[i] - meanmarks2) / (maxmarks2 - minmarks2), marks1sq[i], marks2sq[i], marks1marks2[i]])
		Y_train.append(Y[i])

	for i in range(int(0.7 * len(input_data)), len(input_data)):
		X_test.append([(marks1[i] - meanmarks1) / (maxmarks1 - minmarks1), (marks2[i] - meanmarks2) / (maxmarks2 - minmarks2), marks1sq[i], marks2sq[i], marks1marks2[i]])
		Y_test.append(Y[i])
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
  return ((1 / (((2 * pi) ** (n / 2)) * math.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5 * np.dot(np.transpose(x - mu), np.dot(np.linalg.inv(sigma), (x - mu)))))

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
		if px0_0 >= px0_1:
			if Y_test[i] == 0:
				corr += 1
		else:
			if Y_test[i] == 1:
				corr += 1
	print(corr / len(Y_test) * 100)

X_train, X_test, Y_train, Y_test = getscaleddata()
phy = getphy(Y_train)
mu0, mu1 = getmus(X_train, Y_train)
sigma = getsigma(X_train)
getaccuracy(X_test, Y_test)