import numpy as np
import pandas as pd
import random
import math

input_data = pd.read_csv("Housing Price data set.csv")

Price = input_data['price']
FloorArea = input_data['lotsize']
NoOfBedrooms = input_data['bedrooms']
NoOfBathrooms = input_data['bathrms']

# Performing feature scanning on FloorArea
FloorArea_Mean = np.mean(FloorArea)
FloorArea_Max = max(FloorArea)
FloorArea_Min = min(FloorArea)
FloorArea_Scaled = []
for i in FloorArea:
	FloorArea_Scaled.append((i - FloorArea_Mean) / (FloorArea_Max - FloorArea_Min))

#segmenting the features
FloorAreaTrain = FloorArea[:383]
NoOfBathroomsTrain = NoOfBathrooms[:383]
NoOfBedroomsTrain = NoOfBedrooms[:383]
PriceTrain = []
PriceTrain = Price[:383]

# Function to calculate Slope to find coefficients
def Slope(Coeff, FeaturesTrain, PriceTrain, ind):
	Error = 0
	for i in range(len(FeaturesTrain)):
		itr = 0
		for j in range(len(Coeff)):
			itr = itr + Coeff[j] * FeaturesTrain[i][j]
		Error += (itr - PriceTrain[i]) * FeaturesTrain[i][ind]
	return Error

# Using batch gradient without feature scaling
print("Using batch gradient without feature scaling")
FeaturesTrain = []
for i in range(383):
	FeaturesTrain.append([1, FloorArea[i], NoOfBedrooms[i], NoOfBathrooms[i]])
LearningRateNoScaling = 0.00000001

Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)
lis1 = []
for i in range(5000):
	TempCoeff = Coeff.copy()
	for j in range(len(Coeff)):
		TempCoeff[j] = TempCoeff[j] - ((LearningRateNoScaling / len(FeaturesTrain)) * (Slope(Coeff, FeaturesTrain, PriceTrain, j)))
	Coeff = TempCoeff.copy()

print("Final coefficients are:")
print(Coeff)

PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
	FeaturesTest.append([1, FloorArea[i], NoOfBedrooms[i], NoOfBathrooms[i]])
	PriceTest.append(Price[i])

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * FeaturesTest[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))

# Using batch gradient with feature scaling
print("Using batch gradient with feature scaling")
FeaturesTrain = []
for i in range(383):
	FeaturesTrain.append([1, FloorArea_Scaled[i], NoOfBedrooms[i], NoOfBathrooms[i]])
LearningRateNoScaling = 0.001

Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)
for i in range(5000):
	TempCoeff = Coeff.copy()
	for j in range(len(Coeff)):
		TempCoeff[j] = TempCoeff[j] - ((LearningRateNoScaling / len(FeaturesTrain)) * (Slope(Coeff, FeaturesTrain, PriceTrain, j)))
	Coeff = TempCoeff.copy()

print("Final coefficients are:")
print(Coeff)

PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
	FeaturesTest.append([1, FloorArea_Scaled[i], NoOfBedrooms[i], NoOfBathrooms[i]])
	PriceTest.append(Price[i])

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * FeaturesTest[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))

def SlopeStoch(Coeff,FeaturesTrain,ActualVal,ind):
	itr = 0
	for j in range(len(Coeff)):
		itr = itr + Coeff[j]*FeaturesTrain[j]
	return (itr - ActualVal) * FeaturesTrain[ind]

# Using Stochastic gradient without feature scaling
print("Using Stochastic gradient without feature scaling")

FeaturesTrain = []
for i in range(383):
	FeaturesTrain.append([1, FloorArea[i], NoOfBedrooms[i], NoOfBathrooms[i]])

LearningRateNoScaling = 0.0000000003
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)

for iter in range(10):
	for i in range(len(PriceTrain)):
		TempCoeff = Coeff.copy()
		for j in range(4):
			TempCoeff[j] = TempCoeff[j] - (LearningRateNoScaling * (SlopeStoch(Coeff, FeaturesTrain[i], PriceTrain[i], j)))
		Coeff = TempCoeff.copy()

print("Final coefficients are:")
print(Coeff)

PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
	FeaturesTest.append([1, FloorArea[i], NoOfBedrooms[i], NoOfBathrooms[i]])
	PriceTest.append(Price[i])

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * FeaturesTest[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))


# Using Stochastic gradient with feature scaling
print("Using Stochastic gradient with feature scaling")

FeaturesTrain = []
for i in range(383):
	FeaturesTrain.append([1, FloorArea_Scaled[i], NoOfBedrooms[i], NoOfBathrooms[i]])

LearningRateScaling = 0.005
Coeff = [0, 0, 0, 0]
print("Initial coefficients: ")
print(Coeff)

for iter in range(10):
	for i in range(len(PriceTrain)):
		TempCoeff = Coeff.copy()
		for j in range(4):
			TempCoeff[j] = TempCoeff[j] - (LearningRateScaling * (SlopeStoch(Coeff, FeaturesTrain[i], PriceTrain[i], j)))
		Coeff = TempCoeff.copy()

print("Final coefficients are:")
print(Coeff)

PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
	FeaturesTest.append([1, FloorArea_Scaled[i], NoOfBedrooms[i], NoOfBathrooms[i]])
	PriceTest.append(Price[i])

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * FeaturesTest[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))

# Using Minibatch gradient without feature scaling for batch size = 20
print("Using Minibatch gradient without feature scaling for batch size = 20")
FeaturesTrain = []
for i in range(383):
	FeaturesTrain.append([1, FloorArea[i], NoOfBedrooms[i], NoOfBathrooms[i]])

BatchSize = 20;
LearningRateScaling = 0.000000001
Coeff = [0, 0, 0, 0]
NoOfBatches = math.ceil(len(PriceTrain) / BatchSize)
equallyDiv = False
if (len(PriceTrain) % BatchSize == 0):
	equallyDiv = True;

for epoch in range(30):
	for batch in range(NoOfBatches):
		Summation = [0, 0, 0, 0]
		for j in range(len(Coeff)):
			for i in range(BatchSize):
				if (batch * BatchSize + i == len(FeaturesTrain)):
					break
				PredictedValue = 0.0
				for wj in range(len(Coeff)):
					PredictedValue += Coeff[wj] * FeaturesTrain[batch * BatchSize + i][wj]
				PredictedValue -= PriceTrain[batch * BatchSize + i]
				PredictedValue *= FeaturesTrain[batch * BatchSize + i][j]
				Summation[j] += PredictedValue;

		if (not equallyDiv and batch == NoOfBatches - 1):
			for j in range(len(Summation)):
				Coeff[j] -= (Summation[j] / (len(PriceTrain) % BatchSize)) * LearningRateScaling
		else:
			for j in range(len(Summation)):
				Coeff[j] -= (Summation[j] / BatchSize) * LearningRateScaling
print("Final coefficients are:")
print(Coeff)

PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
	FeaturesTest.append([1, FloorArea[i], NoOfBedrooms[i], NoOfBathrooms[i]])
	PriceTest.append(Price[i])

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * FeaturesTest[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))


# Using Minibatch gradient with feature scaling for batch size = 20
print("Using Minibatch gradient with feature scaling for batch size = 20")

FeaturesTrain = []
for i in range(383):
	FeaturesTrain.append([1, FloorArea_Scaled[i], NoOfBedrooms[i], NoOfBathrooms[i]])

BatchSize = 20;
LearningRateScaling = 0.002
Coeff = [0, 0, 0, 0]
NoOfBatches = math.ceil(len(PriceTrain) / BatchSize)
equallyDiv = False
if (len(PriceTrain) % BatchSize == 0):
	equallyDiv = True;

for epoch in range(30):
	for batch in range(NoOfBatches):
		Summation = [0, 0, 0, 0]
		for j in range(len(Coeff)):
			for i in range(BatchSize):
				if (batch * BatchSize + i == len(FeaturesTrain)):
					break
				PredictedValue = 0.0
				for wj in range(len(Coeff)):
					PredictedValue += Coeff[wj] * FeaturesTrain[batch * BatchSize + i][wj]
				PredictedValue -= PriceTrain[batch * BatchSize + i]
				PredictedValue *= FeaturesTrain[batch * BatchSize + i][j]
				Summation[j] += PredictedValue;

		if (not equallyDiv and batch == NoOfBatches - 1):
			for j in range(len(Summation)):
				Coeff[j] -= (Summation[j] / (len(PriceTrain) % BatchSize)) * LearningRateScaling
		else:
			for j in range(len(Summation)):
				Coeff[j] -= (Summation[j] / BatchSize) * LearningRateScaling
print("Final coefficients are:")
print(Coeff)

PriceTest = []
FeaturesTest = []
for i in range(383, len(Price)):
	FeaturesTest.append([1, FloorArea_Scaled[i], NoOfBedrooms[i], NoOfBathrooms[i]])
	PriceTest.append(Price[i])

# Finding Mean absolute percentage error.
Error = 0
for i in range(len(FeaturesTest)):
	predicted = 0
	for j in range(len(Coeff)):
	  	predicted = predicted + Coeff[j] * FeaturesTest[i][j]
	Error += abs(predicted - PriceTest[i]) / PriceTest[i]
Error = (Error / len(FeaturesTest)) * 100
print("Mean absolute percentage error is : " + str(Error))