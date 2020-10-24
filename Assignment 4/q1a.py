# In this code I have implemented logistic regression using batch gradient, stochaistic gradient and mini batch gradient for feature scaled and unscaled data.

import numpy as np
import pandas as pd
import random
import math

def getdata():
	input_data = pd.read_csv("marks.csv")
	Y = input_data['selected']
	marks1 = input_data['marks1']
	marks2 = input_data['marks2']
	
	X_train = []
	X_test = []
	Y_train = []
	Y_test = []
	for i in range(70):
		X_train.append([1, marks1[i], marks2[i]])
		Y_train.append(Y[i])

	for i in range(71, 100):
		X_test.append([1, marks1[i], marks2[i]])
		Y_test.append(Y[i])
	return X_train, X_test, Y_train, Y_test

def getscaleddata():
	input_data = pd.read_csv("marks.csv")
	Y = input_data['selected']
	marks1 = input_data['marks1']
	marks2 = input_data['marks2']

	meanmarks1 = np.mean(marks1)
	maxmarks1 = np.max(marks1)
	minmarks1 = np.min(marks1)

	meanmarks2 = np.mean(marks2)
	maxmarks2 = np.max(marks2)
	minmarks2 = np.min(marks2)

	X_train = []
	X_test = []
	Y_train = []
	Y_test = []

	for i in range(70):
		X_train.append([1, (marks1[i] - meanmarks1) / (maxmarks1 - minmarks1), (marks2[i] - meanmarks2) / (maxmarks2 - minmarks2)])
		Y_train.append(Y[i])

	for i in range(70, 100):
		X_test.append([1, (marks1[i] - meanmarks1) / (maxmarks1 - minmarks1), (marks2[i] - meanmarks2) / (maxmarks2 - minmarks2)])
		Y_test.append(Y[i])
	return X_train, X_test, Y_train, Y_test

def sigmoid(z):
    return 1.0 / (1 + math.exp(-1 * z))

# Function to calculate Slope to find coefficients
def Slope(Coeff, X_train, Y_train, ind):
	diff = 0
	for i in range(len(X_train)):
		itr = 0
		for j in range(len(Coeff)):
			itr = itr + Coeff[j] * X_train[i][j]
		diff += (sigmoid(itr) - Y_train[i]) * X_train[i][ind]
	return diff

# Using batch gradient
def batchgra(X_train, Y_train, alpha = 0.00001, epochs = 50000):
	LearningRateNoScaling = alpha

	Coeff = [0, 0, 0]
	lis1 = []
	for i in range(epochs):
		TempCoeff = Coeff.copy()
		for j in range(len(Coeff)):
			TempCoeff[j] = TempCoeff[j] - ((LearningRateNoScaling / len(X_train)) * (Slope(Coeff, X_train, Y_train, j)))
		Coeff = TempCoeff.copy()
	return Coeff

# Finding Accuracy
def printaccuracy(X_test, Y_test, Coeff):
	count = 0
	for i in range(len(X_test)):
		predicted = 0
		for j in range(len(Coeff)):
		  	predicted = predicted + Coeff[j] * X_test[i][j]
		predicted = sigmoid(predicted)
		if predicted > 0.5:
			if Y_test[i] == 1:
				count += 1
		else:
			if Y_test[i] == 0:
				count += 1
	print("Accuracy is : " + str(count / len(Y_test) * 100))

def SlopeStoch(Coeff, X_train, ActualVal, ind):
	itr = 0
	for j in range(len(Coeff)):
		itr = itr + Coeff[j] * X_train[j]
	return (sigmoid(itr) - ActualVal) * X_train[ind]

def stochgra(X_train, Y_train, alpha = 0.00001, epochs = 50000):
	LearningRateNoScaling = alpha
	Coeff = [0, 0, 0]
	for iter in range(epochs):
		for i in range(len(Y_train)):
			TempCoeff = Coeff.copy()
			for j in range(3):
				TempCoeff[j] = TempCoeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train[i], Y_train[i], j)))
			Coeff = TempCoeff.copy()
	return Coeff

def minibtchgra(X_train, Y_train, alpha = 0.000000001, epochs = 30, batchsize = 20):
	LearningRateScaling = alpha
	Coeff = [0, 0, 0]
	NoOfBatches = math.ceil(len(Y_train) / batchsize)
	equallyDiv = False
	if (len(Y_train) % batchsize == 0):
		equallyDiv = True;

	for epoch in range(epochs):
		for batch in range(NoOfBatches):
			Summation = [0, 0, 0]
			for j in range(len(Coeff)):
				for i in range(batchsize):
					if (batch * batchsize + i == len(X_train)):
						break
					PredictedValue = 0.0
					for wj in range(len(Coeff)):
						PredictedValue += Coeff[wj] * X_train[batch * batchsize + i][wj]
					PredictedValue = sigmoid(PredictedValue)
					PredictedValue -= Y_train[batch * batchsize + i]
					PredictedValue *= X_train[batch * batchsize + i][j]
					Summation[j] += PredictedValue;

			if (not equallyDiv and batch == NoOfBatches - 1):
				for j in range(len(Summation)):
					Coeff[j] -= (Summation[j] / (len(Y_train) % batchsize)) * LearningRateScaling
			else:
				for j in range(len(Summation)):
					Coeff[j] -= (Summation[j] / batchsize) * LearningRateScaling
	return Coeff

# First doing batch gradient, stochaistic gradient and mini batch gradient without feature scaling.
X_train, X_test, Y_train, Y_test = getdata()

print("Doing batch gradient without feature scaling")
coeff = batchgra(X_train, Y_train, 0.00001, 5000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing stochaistic gradient without feature scaling")
coeff = stochgra(X_train, Y_train, 0.001, 5000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing Mini batch gradient without feature scaling")
coeff = minibtchgra(X_train, Y_train, 0.0001, 100, 20)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

# Now doing batch gradient, stochaistic gradient and mini batch gradient with feature scaling.
X_train, X_test, Y_train, Y_test = getscaleddata()

print("Doing batch gradient with feature scaling")
coeff = batchgra(X_train, Y_train, 0.00001, 5000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing stochaistic gradient with feature scaling")
coeff = stochgra(X_train, Y_train, 0.001, 5000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing Mini batch gradient with feature scaling")
coeff = minibtchgra(X_train, Y_train, 0.0001, 100, 20)
print(coeff)
printaccuracy(X_test, Y_test, coeff)