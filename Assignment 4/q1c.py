# In this code I have implemented feature scaled logistic regression using batch gradient, stochaistic gradient and mini batch gradient with and without regularisation.
# Here I have also used higher powers of the data to make more features.
# Now the hypothesis looks like h(x) = g(wx) where g(wx) = 1 / (1 + e^(-wx)) and wx = w0 + w1x + w2y + w3x^2 + w4y^2 + w5xy + w6x^3 + w7y^3 + w8x^2y + w9xy^2
import numpy as np
import pandas as pd
import random
import math

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

	marks1cu = []
	for i in marks1:
		marks1cu.append(i * i * i)
	meanmarks1cu = np.mean(marks1cu)
	maxmarks1cu = np.max(marks1cu)
	minmarks1cu = np.min(marks1cu)
	for i in range(len(marks1cu)):
		marks1cu[i] = (marks1cu[i] - meanmarks1cu) / (maxmarks1cu - minmarks1cu)

	marks2cu = []
	for i in marks2:
		marks2cu.append(i * i * i)
	meanmarks2cu = np.mean(marks2cu)
	maxmarks2cu = np.max(marks2cu)
	minmarks2cu = np.min(marks2cu)
	for i in range(len(marks2cu)):
		marks2cu[i] = (marks2cu[i] - meanmarks2cu) / (maxmarks2cu - minmarks2cu)

	marks1sqmarks2 = []
	for i in range(len(marks1)):
		marks1sqmarks2.append(marks1[i] * marks1[i] * marks2[i])
	meanmarks1sqmarks2 = np.mean(marks1sqmarks2)
	maxmarks1sqmarks2 = np.max(marks1sqmarks2)
	minmarks1sqmarks2 = np.min(marks1sqmarks2)
	for i in range(len(marks1sqmarks2)):
		marks1sqmarks2[i] = (marks1sqmarks2[i] - meanmarks1sqmarks2) / (maxmarks1sqmarks2 - minmarks1sqmarks2)

	marks2sqmarks1 = []
	for i in range(len(marks1)):
		marks2sqmarks1.append(marks1[i] * marks2[i] * marks2[i])
	meanmarks2sqmarks1 = np.mean(marks2sqmarks1)
	maxmarks2sqmarks1 = np.max(marks2sqmarks1)
	minmarks2sqmarks1 = np.min(marks2sqmarks1)
	for i in range(len(marks2sqmarks1)):
		marks2sqmarks1[i] = (marks2sqmarks1[i] - meanmarks2sqmarks1) / (maxmarks2sqmarks1 - minmarks2sqmarks1)

	X_train = []
	X_test = []
	Y_train = []
	Y_test = []

	for i in range(70):
		X_train.append([1, (marks1[i] - meanmarks1) / (maxmarks1 - minmarks1), (marks2[i] - meanmarks2) / (maxmarks2 - minmarks2), marks1sq[i], marks2sq[i], marks1marks2[i], marks1cu[i], marks2cu[i], marks1sqmarks2[i], marks2sqmarks1[i]])
		Y_train.append(Y[i])

	for i in range(71, 100):
		X_test.append([1, (marks1[i] - meanmarks1) / (maxmarks1 - minmarks1), (marks2[i] - meanmarks2) / (maxmarks2 - minmarks2), marks1sq[i], marks2sq[i], marks1marks2[i], marks1cu[i], marks2cu[i], marks1sqmarks2[i], marks2sqmarks1[i]])
		Y_test.append(Y[i])
	return X_train, X_test, Y_train, Y_test

def sigmoid(z):
	try:
		ans = 1.0 / (1 + math.exp(-1 * z))
	except OverflowError:
		ans = 0
	return ans

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
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for iter in range(epochs):
		for i in range(len(Y_train)):
			TempCoeff = Coeff.copy()
			for j in range(len(Coeff)):
				TempCoeff[j] = TempCoeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train[i], Y_train[i], j)))
			Coeff = TempCoeff.copy()
	return Coeff

def minibtchgra(X_train, Y_train, alpha = 0.000000001, epochs = 30, batchsize = 20):
	LearningRateScaling = alpha
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	NoOfBatches = math.ceil(len(Y_train) / batchsize)
	equallyDiv = False
	if (len(Y_train) % batchsize == 0):
		equallyDiv = True;

	for epoch in range(epochs):
		for batch in range(NoOfBatches):
			Summation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

# Using batch gradient
def batchgrareg(X_train, Y_train, alpha = 0.00001, epochs = 50000, lambdaparameter = -49):
	LearningRateNoScaling = alpha

	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	lis1 = []
	for i in range(epochs):
		TempCoeff = Coeff.copy()
		for j in range(len(Coeff)):
			if j == 0:
				TempCoeff[j] = TempCoeff[j] - ((LearningRateNoScaling / len(X_train)) * (Slope(Coeff, X_train, Y_train, j)))
			else:
				TempCoeff[j] = (1 - alpha * lambdaparameter / len(X_train)) * TempCoeff[j] - ((LearningRateNoScaling / len(X_train)) * (Slope(Coeff, X_train, Y_train, j)))
		Coeff = TempCoeff.copy()
	return Coeff

def stochgrareg(X_train, Y_train, alpha = 0.00001, epochs = 50000, lambdaparameter = 1000):
	LearningRateNoScaling = alpha
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	for iter in range(epochs):
		for i in range(len(Y_train)):
			TempCoeff = Coeff.copy()
			for j in range(len(Coeff)):
				if j == 0:
					TempCoeff[j] = TempCoeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train[i], Y_train[i], j)))
				else:
					TempCoeff[j] = (1 - alpha * lambdaparameter) * TempCoeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, X_train[i], Y_train[i], j)))
			Coeff = TempCoeff.copy()
	return Coeff

def minibtchgrareg(X_train, Y_train, alpha = 0.000000001, epochs = 30, batchsize = 20, LambdaParameter = 10):
	LearningRateScaling = alpha
	Coeff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	NoOfBatches = math.ceil(len(Y_train) / batchsize)
	equallyDiv = False
	if (len(Y_train) % batchsize == 0):
		equallyDiv = True;

	for epoch in range(epochs):
		for batch in range(NoOfBatches):
			Summation = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
					if j == 0:
						Coeff[j] = Coeff[j] - (Summation[j] / (len(Y_train) % batchsize)) * LearningRateScaling
					else:
						Coeff[j] = (1 - LearningRateScaling * LambdaParameter / (len(Y_test) % batchsize)) * Coeff[j] - (Summation[j] / (len(Y_train) % batchsize)) * LearningRateScaling
			else:
				for j in range(len(Summation)):
					if j == 0:
						Coeff[j] = Coeff[j] - (Summation[j] / batchsize) * LearningRateScaling
					else:
						Coeff[j] = (1 - LearningRateScaling * LambdaParameter / batchsize) * Coeff[j] - (Summation[j] / batchsize) * LearningRateScaling
	return Coeff

# First doing batch gradient, stochaistic gradient and mini batch gradient without regularisation.
X_train, X_test, Y_train, Y_test = getscaleddata()

print("Doing batch gradient without regularisation")
coeff = batchgra(X_train, Y_train, 0.00001, 1000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing stochaistic gradient without regularisation")
coeff = stochgra(X_train, Y_train, 0.0001, 5000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing Mini batch gradient without regularisation")
coeff = minibtchgra(X_train, Y_train, 0.0001, 100, 32)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

Now doing batch gradient, stochaistic gradient and mini batch gradient with regularisation.
print("Doing batch gradient with regularisation")
coeff = batchgrareg(X_train, Y_train, 0.0001, 5000, 1000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing stochaistic gradient with regularisation")
coeff = stochgrareg(X_train, Y_train, 0.001, 500, 1000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)

print("Doing Mini batch gradient with regularisation")
coeff = minibtchgrareg(X_train, Y_train, 0.0001, 1000, 32, 1000)
print(coeff)
printaccuracy(X_test, Y_test, coeff)