# Code by Prateek Mishra, IIT2018199 IIIT-Allahabad
'''
In this code I have implemented facial recognition using PCA.
input :
	root - This is the root directory of the project.
	root - The root directory of the project.
	ratio - The ratio of test and train split.
	varients - The number of varients in the dataset
	imagesInEachVarient - This number of samples of each varient
	totalPixels - The number of pixels in each image of the 
	dataset(Image Width * Image Length)
Output :
1. Calculates the percentage accuracy in prediction.
2. Identifies the imposter.
'''
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

def getFaceVector(root, ratio = 0.6,
 variants = 40, imagesInEachVarient = 10, totalPixels = 92 * 112):
	'''
	Input:
		root - The root directory of the project.
		ratio - The ratio of test and train split.
		varients - The number of varients in the dataset
		imagesInEachVarient - This number of samples of each varient
		totalPixels - The number of pixels in each image of the 
		dataset(Image Width * Image Length)
	Desc:
		Converts the face images to face vectors
	Output:
		The calculated faceVector
	'''
	faceVector = []
	for i in range(1, variants + 1):
		for img in range (1, int(ratio * imagesInEachVarient) + 1):
			image = root + "dataset/s" + str(i) + "/" + str(img) + ".pgm"
			inputImage = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)
			faceVector.append(inputImage.reshape(totalPixels,))
	return np.asarray(faceVector).transpose()

def getEigenVectors(faceVector):
	'''
	Input:
		faceVector - The face vectors calculated from the training database.
	Desc:
		Normalises the face vectors by calculating the average face vector 
		and then subtracting it from each vector. Then calculates the Sigma.
		Using sigma then it calculates the eigen vectors.
	Output:
		eigenValues - The calculated eigen values.
		eigenVectors - The calculated eigen vectors.
		meanFaceVector - The calculated average face vector
		normalised_faceVector - The difference of face vector and the 
		meanFaceVector.
	'''
	meanFaceVector = faceVector.mean(axis=1).reshape(faceVector.shape[0], 1)
	norFaceVector = faceVector - meanFaceVector
	eigenValues, eigenVectors = np.linalg.eig(np.cov(np.transpose(norFaceVector)))
	return eigenValues, eigenVectors, meanFaceVector, norFaceVector

def getKEigenVectors(eigenVectors, k = 20):
	'''
	Input:
		eigenVectors - The calculated eigen vectors.
		k - The k parameter to calculate top k values.
	Desc:
		Selects the K best Eigen Faces, K < M. Then converts the lower
		dimensionality K eigen Vectors to Original Dimensionality.
	Output:
		weights - The eigen faces converted to original dimensionality.
		eigenFaces - The top k eigen vector dot with transpose of 
		normalized face vector
	'''
	kEigenVectors = eigenVectors[0:k, :]
	eigenFaces = kEigenVectors.dot(np.transpose(norFaceVector))
	weights = np.transpose(norFaceVector).dot(np.transpose(eigenFaces))
	return weights, eigenFaces

def getAccuracy(weights, meanFaceVector, eigenFaces):
	'''
	Input:
		weights - The eigen faces converted to original dimensionality.
		meanFaceVector - The calculated average face vector
		eigenFaces - The top k eigen vector dot with transpose of 
		normalized face vector
	Desc:
		This is the testing function it loads up all images from 
		the testing dataset then matches it with each and every image of 
		the traning dataset. Finally one with the minimum distance is
		the class it is put in. Then based on all these predictions it
		calculates the accuracy of the algorithm.
		Parallely it also computes the maximum possible minimum distance
		for the training data.
	Output:
		accuracy - The accuracy in doing facial recognition on the
		dataset.
		maxw - The maximum distance that is possible on the given dataset.
	'''
	total_images = 160
	maxw = -1
	correct = 0
	for i in range(1, variants + 1):
		for img in range (int(ratio * imagesInEachVarient) + 1, imagesInEachVarient + 1):
			image = root + "dataset/s" + str(i) + "/" + str(img) + ".pgm"
			test_img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2GRAY)
			test_img = test_img.reshape(totalPixels, 1)
			test_norFaceVector = test_img - meanFaceVector
			test_weight = np.transpose(test_norFaceVector).dot(np.transpose(eigenFaces))
			index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))
			weight = np.min(np.linalg.norm(test_weight - weights, axis=1))
			maxw = max(maxw, weight)
			if int(index / 6) + 1 == i:
				correct += 1
	return (correct / total_images * 100), maxw + 1

def isImposter(path, maxw, meanFaceVector, weights):
	'''
	Input:
		path - The path of the imposter image.
		maxw - The maximum distance that is possible on the given dataset.
		meanFaceVector - The calculated average face vector
		weights - The eigen faces converted to original dimensionality.
	Desc:
		This is the testing function it loads up the imposter image from 
		the dataset then matches it with each and every image of 
		the traning dataset. Finally one with the minimum distance is
		calculated. If this minimum distance is greater than the maximum
		distance possible then this is the imposter.
	Output:
		Imposter - If the sample is an imposter.
		Not Imposter - If the sample is not an imposter.
	'''
	test_img = cv2.cvtColor(cv2.imread(root + "dataset/imposter.png"), cv2.COLOR_RGB2GRAY)
	test_img = test_img.reshape(totalPixels, 1)
	test_norFaceVector = test_img - meanFaceVector
	test_weight = np.transpose(test_norFaceVector).dot(np.transpose(eigenFaces))
	weight = np.min(np.linalg.norm(test_weight - weights, axis=1))
	index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))
	if(weight > maxw):
		return 'Yes! Imposter'
	else:
		return 'No! Not an Imposter'

# These are the dataset properties.
root = '/home/kinshuu/OneDrive/Documents/Content Sem 5/IML/Assignments/Assignment 8 (b)/'
ratio = 0.6
variants = 40
imagesInEachVarient = 10
totalPixels = 92 * 112 # image width * image height

x = []
y = []
maxAccuracy = -1
minAccuracy = 101

for i in range(1, 160):
	faceVector = getFaceVector(root, ratio, variants, imagesInEachVarient, totalPixels)
	eigenValues, eigenVectors, meanFaceVector, norFaceVector = getEigenVectors(faceVector)
	weights, eigenFaces = getKEigenVectors(eigenVectors, k = i)
	accuracy, maxw = getAccuracy(weights, meanFaceVector, eigenFaces)
	x.append(i)
	y.append(accuracy)
	maxAccuracy = max(maxAccuracy, accuracy)
	minAccuracy = min(minAccuracy, accuracy)

# plotting the points  
plt.plot(x, y)
plt.xlabel('K values')
plt.ylabel('Accuracies')
plt.title('Top K Eigen Faces VS Accuracies')
plt.show()

print("Min Accuracy is : " + str(minAccuracy))
print("Max Accuracy is : " + str(maxAccuracy))

# The path of the imposter image.
path = root + "dataset/imposter.png"
print(isImposter(path, maxw, meanFaceVector, weights))

path = root + "dataset/imposter2.png"
print(isImposter(path, maxw, meanFaceVector, weights))