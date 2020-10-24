# In this code I have predicted heart diseases using the clevland medical data. I have used mini batch GDA on regularised and feature scaled data.
import numpy as np
import pandas as pd
import random
import math

def getscaleddata():
	input_data = pd.read_csv("heart_diseases.csv")
	n = input_data.shape[0]
	Y = input_data['y']
	f1 = input_data['f1']
	f1 = (f1 - np.mean(f1)) / (np.max(f1) - np.min(f1))
	f2 = input_data['f2']
	f2 = (f2 - np.mean(f2)) / (np.max(f2) - np.min(f2))
	f3 = input_data['f3']
	f3 = (f3 - np.mean(f3)) / (np.max(f3) - np.min(f3))
	f4 = input_data['f4']
	f4 = (f4 - np.mean(f4)) / (np.max(f4) - np.min(f4))
	f5 = input_data['f5']
	f5 = (f5 - np.mean(f5)) / (np.max(f5) - np.min(f5))
	f6 = input_data['f6']
	f6 = (f6 - np.mean(f6)) / (np.max(f6) - np.min(f6))
	f7 = input_data['f7']
	f7 = (f7 - np.mean(f7)) / (np.max(f7) - np.min(f7))
	f8 = input_data['f8']
	f8 = (f8 - np.mean(f8)) / (np.max(f8) - np.min(f8))
	f9 = input_data['f9']
	f9 = (f9 - np.mean(f9)) / (np.max(f9) - np.min(f9))
	f10 = input_data['f10']
	f10 = (f10 - np.mean(f10)) / (np.max(f10) - np.min(f10))
	f11 = input_data['f11']
	f11 = (f11 - np.mean(f11)) / (np.max(f11) - np.min(f11))
	f12 = input_data['f12']
	f12 = (f12 - np.mean(f12)) / (np.max(f12) - np.min(f12))
	f13 = input_data['f13']
	f13 = (f13 - np.mean(f13)) / (np.max(f13) - np.min(f13))
	X_train = []
	X_test = []
	Y_train = []
	Y_test = []
	for i in range(int(0.7 * n)):
		X_train.append([1, f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i], f9[i], f10[i], f11[i], f12[i], f13[i]])
		Y_train.append(Y[i])

	for i in range(int(0.7 * n), n):
		X_test.append([1, f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i], f9[i], f10[i], f11[i], f12[i], f13[i]])
		Y_test.append(Y[i])
	return X_train, X_test, Y_train, Y_test

def sigmoid(z):
    return 1.0 / (1 + math.exp(-1 * z))

def minibtchgrareg(X_train, Y_train, alpha = 0.000000001, epochs = 30, batchsize = 20, LambdaParameter = 10):
	LearningRateScaling = alpha
	Coeff = [0] * len(X_train[0])
	NoOfBatches = math.ceil(len(Y_train) / batchsize)
	equallyDiv = False
	if (len(Y_train) % batchsize == 0):
		equallyDiv = True;

	for epoch in range(epochs):
		for batch in range(NoOfBatches):
			Summation = [0] * len(X_train[0])
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

def getconfusionmat(X_test, Y_test, Coeff):
	truepositives = 0
	falsepositives = 0
	truenegatives = 0
	falsenegatives = 0
	for i in range(len(X_test)):
		predicted = 0
		for j in range(len(Coeff)):
		  	predicted = predicted + Coeff[j] * X_test[i][j]
		predicted = sigmoid(predicted)
		if predicted > 0.5:
			if Y_test[i] == 1:
				truepositives += 1
			else:
				falsepositives += 1
		else:
			if Y_test[i] == 0:
				truenegatives += 1
			else:
				falsenegatives += 1
	predictedpositive = []
	predictednegative = []
	confustionmatrix = []
	predictedpositive.append(truepositives)
	predictedpositive.append(falsepositives)
	predictednegative.append(falsenegatives)
	predictednegative.append(truenegatives)
	confustionmatrix.append(predictedpositive)
	confustionmatrix.append(predictednegative)
	return confustionmatrix

X_train, X_test, Y_train, Y_test = getscaleddata()
print("Doing Mini batch gradient with feature scaling and regularised data")
coeff = minibtchgrareg(X_train, Y_train, 0.00001, 500, 64, 10)
print("Final coefficients are : ")
print(coeff)
printaccuracy(X_test, Y_test, coeff)
print("Now printing the confustion matrix")
print(getconfusionmat(X_test, Y_test, coeff))