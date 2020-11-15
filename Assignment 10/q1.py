# Code by Prateek Mishra, IIT2018199
'''
In this code I have implemented a Kohonen network with 100 neurons arranged in 
the form of a two-dimensional lattice with 10 rows and 10 columns. I have trained
the network with 1500 1500 two-dimensional input vectors generated randomly in
a square region in the interval between -1 and +1. Finally I have tested the 
performance with the following input vectors:
Test the performance of the self organizing neurons using the following
Input vectors:
X1=[0.1 0.8], X2=[0.5 -0.2], X3=[-0.8 -0.9], X4=[-0.06 0.9]
'''

import numpy as np

def getData():
    '''
    Desc : This fucntion calculates 1500 random vectors for the traning data.
    Return : This function returns the data for network traning and the data
    network uses for testing.
    '''
    Xtrain = 2 * np.random.rand(1500, 2) - 1
    Xtest = [[0.1, 0.8], [0.5, -0.2], [-0.8, -0.9], [-0.06, 0.9]]
    return Xtrain, Xtest

def getAlpha(ind, alpha, lambd):
    '''
    Input : 
        ind - The iteration number in the training process.
        alpha - The alpha model parameter
        lambd - The lambd model parameter
    Desc :
        This function return the learning parameter, decreasing
        it with time.
    Return : 
        The discounted learning parameter.
    '''
    return alpha * np.exp(-ind / lambd) 

def getPenalty(bmuDistance, ind, sigma, lambd):
    '''
    Input :
        bmuDistance - This is the best matching unit distance.
        ind - The iteration number in the training process.
        sigma - The sigma model parameter
        lambd - The lambd model parameter
    Desc :
        This function calculates the neighbouring penalty
    Return : 
        Returns the calculated neighbouring penalty.
    '''
    sigmaT = getSigma(ind, sigma, lambd)
    return np.exp(-(bmuDistance * bmuDistance) / (2 * sigmaT * sigmaT))

def getSigma(ind, sigma, lambd):
    '''
    Input : 
        ind - The iteration number in the training process.
        sigma - The sigma model parameter
        lambd - The lambd model parameter
    Desc :
        This function return the sigma parameter, decreasing
        it with time.
    Return : 
        The discounted sigma parameter.
    '''
    return sigma * np.exp(-ind / lambd) 

def getWeights(Xtrain, alpha = 0.1, lambd = 100.0, sigma = 10):
    '''
    Input :
        Xtrain - The training list of vectors.
        alpha - The alpha model parameter.
        sigma - The sigma model parameter.
        lambd - The lambd model parameter.
    Desc :
        This function initialises the weights with random values. Then updates 
        the weights with each iteration till there is a significant spread.
        It first takes a random input vector and then calculates the minimum best
        matching unit for that input vector. Finally it updates the weights using
        the decay rate and distance from the bmu.
    Return : 
        The refined value of weights.
    '''
    rows = 10
    columns = 10
    weights = 2 * np.random.rand(rows, columns, 2) - 1
    for ind in range(Xtrain.shape[0]):
        if getSigma(ind, sigma, lambd) < 0.01:
            break

        inputVector = Xtrain[np.random.choice(range(len(Xtrain)))]
        
        minDist = float('inf')
        for i in range(rows):
            for j in range(columns):
                vecDist = np.linalg.norm((inputVector - weights[i, j]))
                if vecDist < minDist:
                    minDist = vecDist
                    minBMU = (i, j)
        
        for i in range(rows):
            for j in range(columns):
                bmuDistance = np.linalg.norm((np.array(minBMU) - np.array((i, j))))
                weights[i][j] += getPenalty(bmuDistance, ind, sigma, lambd) \
                * getAlpha(ind, alpha, lambd) \
                * (inputVector - weights[i][j])
    return weights

def test(weights, Xtest):
    '''
    Input :
        Weights : This is the weights calculated while model traning.
        Xtest : This is the testing data.
    Desc :
        It calculates the minimum best matching unit for each of the testing input vector.
        For the best matching unit it prints the properties on the screen.
    '''
    rows = weights.shape[0]
    columns = weights.shape[1]
    for inputVector in Xtest:
        minDist = float('inf')
        for i in range(rows):
            for j in range(columns):
                vecDist = np.linalg.norm((inputVector - weights[i, j]))
                if vecDist < minDist:
                    minDist = vecDist
                    minBMU = (i, j)
        print('Input : ' + str(inputVector))
        print('Minimum Euclidian distance : ' + str(minDist))
        print('Cluster : ' + str(minBMU) + '\n')

Xtrain, Xtest = getData()
weights = getWeights(Xtrain, 0.1)
test(weights, Xtest)