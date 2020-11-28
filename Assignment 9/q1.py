#Code by Prateek Mishra, IIT2018199 IIII-Allahabad.
'''
In this code I have implemented a bidirectional assiciative memory with
four pairs of patterns. I have implemented a weight matrix and tested 
the weight of corrections.
'''
import numpy as np

def getdata():
	xAxis = np.matrix([[1, 1, 1, 1, 1, 1 ], [-1, -1, -1, -1, -1, -1 ], 
		[1, -1, -1, 1, 1, 1 ], [1, 1, -1, -1, -1, -1 ]])
	yAxis = np.matrix([[1, 1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1]])
	return xAxis, yAxis

'''
This function performs thresholding and converts input vector to bipolar
form
'''
def getBipolarForm(mat):
    ret = []
    for ele in mat[0]:
        if ele <= 0:
            ret.append(-1)
        else:
            ret.append(1)
    return np.array(ret)

def getCoeff(n = 6, m = 3, M = 4):
	weights = np.zeros((n,m))
	for i in range(M):
	    weights += np.transpose(xAxis[i]).dot(yAxis[i])
	return weights
   
def testCorrectionX(z, layer):
	print("M is : " + str(z + 1))
	print("Input is : ")
	print(layer[z])
	layerDotWeights = layer[z].dot(weights)
	bipolarForm = getBipolarForm(np.array(layerDotWeights))
	print('Bipolar Form :')
	print(bipolarForm)
	print('Targeted output :')
	print(np.asarray(yAxis[z]).reshape(-1))
	if(np.array_equal(bipolarForm, np.asarray(yAxis[z]).reshape(-1))):
		print('Both are equal. Thus verified.')
	else:
		print('Booo! Not equal!')

def testCorrectionY(z, layer):
	print("M is : " + str(z + 1))
	print("Input is : ")
	print(layer[z])
	layerDotWeightsT = yAxis[z].dot(np.transpose(weights))
	bipolarForm = getBipolarForm(np.array(layerDotWeightsT))
	print('Bipolar Form :')
	print(bipolarForm)
	print('Targeted output :')
	print(np.asarray(xAxis[z]).reshape(-1))
	if(np.array_equal(bipolarForm, np.asarray(xAxis[z]).reshape(-1))):
		print('Both are equal. Thus verified.')
	else:
		print('Booo! Not equal!')

def testCorrection(xAxis, yAxis, weights):
	print("\nTesting for xAxis")
	testCorrectionX(0, xAxis)
	print("\nTesting for yAxis")
	testCorrectionY(3, yAxis)

xAxis, yAxis = getdata()
weights = getCoeff()
print("weight matrix is : ")
print(weights)
testCorrection(xAxis, yAxis, weights)