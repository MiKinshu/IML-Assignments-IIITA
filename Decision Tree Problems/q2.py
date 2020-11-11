#Code by Prateek Mishra, IIT2018199 IIIT-Allahabad.
'''
About:
In this code I have implemented a decision tree based on the slope feature of the dataset.
This code finds the best rank (slope here) using information gain to form a 
decision tree of depth 1.

Input:
The input is the heartDisease.csv file.

Output:
Output is the rank (slope here) and the information gain associated with that rank (slope here).

Refrences:
To make this code I took help from the following sources:
1. https://www.youtube.com/watch?v=7VeUPuFGJHk (StatQuest with Josh Starmer)
2. https://drive.google.com/file/d/1s19DQaE7GyGAeYd-HTcAVqfFbux4yRJI/view?usp=sharing (Course Slide)
3. https://www.geeksforgeeks.org/python-pandas-dataframe-sort_values-set-1/ (GeekforGeeks)
'''
import numpy as np
import pandas as pd
import math

def getSortedData():
    '''
    This function reads the data from the dataset file
    and extracts features and sorts them.
    Output : 
        Returns the extracted features.
    '''
    input_data = pd.read_csv("heartDisease.csv")
    input_data = input_data.sample(frac = 1).reset_index(drop = True) #Shuffling the data.
    input_data.sort_values("f1", axis = 0, ascending = True,
                inplace = True, na_position = 'last')
    y = input_data['y']
    f1 = input_data['f11']
    return np.array(y), np.array(f1)

def calculateInformationGain(lessThanYes, lessThanNo, 
    greaterThanYes, greaterThanNo, initialEntropy):
    '''
    Input : 
        lessThanYes : This is the no of samples that belong to the yes category
        on the less than part of the node.
        lessThanNo : This is the no of samples that belong to the no category
        on the less than part of the node.
        greaterThanYes : This is the no of samples that belong to the yes category
        on the greater than part of the node.
        greaterThanNo : This is the no of samples that belong to the no category
        on the greater than part of the node.
    Output : 
        Calculated Information gain.
    About : 
        It calculates the information gain.
    '''
    lessThanTotal = lessThanYes + lessThanNo
    lessThanYesProbability = lessThanYes / lessThanTotal
    lessThanNoProbability = lessThanNo / lessThanTotal

    greaterThanTotal = greaterThanYes + greaterThanNo
    greaterThanYesProbability = greaterThanYes / greaterThanTotal
    greaterThanNoProbability = greaterThanNo / greaterThanTotal

    total = lessThanTotal + greaterThanTotal

    lessThanEntropy = -1 * (lessThanYesProbability * math.log2(lessThanYesProbability) + lessThanNoProbability + math.log2(lessThanNoProbability))
    greaterThanEntropy = -1 * (greaterThanNoProbability * math.log2(greaterThanNoProbability) + greaterThanYesProbability * math.log2(greaterThanYesProbability))
    informationGain = initialEntropy - greaterThanEntropy - lessThanEntropy
    return informationGain

def getInformationGain(slopeRank, f1, y, initialEntropy):
    '''
    Input : 
        slopeRank : The rank about which the information gain is to be calculated
    Output : 
        The information gained returned by the calculateInformationGain helper function.
    About : 
        It calculated the parameters required to calculate the Information gain for 
        the dataset around the slopeRank.
    '''
    lessThanYes, lessThanNo, greaterThanYes, greaterThanNo = 0, 0, 0, 0

    for i in range(len(f1)):
        if(f1[i] <= slopeRank):
            if(y[i] == 1):
                lessThanYes += 1
            else:
                lessThanNo += 1
        else:
            if (y[i] == 1):
                greaterThanYes += 1
            else:
                greaterThanNo += 1
    informationGain = calculateInformationGain(lessThanYes, lessThanNo, 
        greaterThanYes, greaterThanNo, initialEntropy)
    return informationGain

def getBatchEntropy(f1, y):
    '''
    Input : 
        f1 : This is the feature on which the tree node would be made.
        y : This the the feature representing the final output class.
    Output : 
        The Initial Entropy of a batch.
    About : 
        This function iterates through all the values in the f1 feature and
        then calculates the initial Entropy.
    '''
    classOne, classZero = 0, 0
    for i in range(len(f1)):
        if y[i] == 1:
            classOne += 1
        else:
            classZero += 1
    total = classOne + classZero
    if total == 0:
        return 0
    initialEntropy = (classZero / total) * math.log2(classZero / total ) + (classOne / total) * math.log2(classOne / total)
    return -initialEntropy

def getBestSlopeRank(f1, y):
    '''
    Input : 
        f1 : This is the feature on which the tree node would be made.
    Output : 
        The best rank around which the decision tree node should be made.
    About : 
        This function iterates through all the values in the f1 feature and
        then chooses the one value which has the maximum Information gain.
    '''
    initialEntropy = getBatchEntropy(f1, y)
    uniqueF = list(set(f1))
    bestSlopeRank = (uniqueF[0] + uniqueF[1]) / 2.0
    maxInformationGain = getInformationGain(bestSlopeRank, f1, y, initialEntropy)
    for i in range(len(uniqueF) - 1):
        slopeRank = (uniqueF[i] + uniqueF[i + 1]) / 2.0
        temp = getInformationGain(slopeRank, f1, y, initialEntropy)
        if(temp > maxInformationGain):
            maxInformationGain = temp
            bestSlopeRank = slopeRank
    return bestSlopeRank, maxInformationGain

def partition(f1, y, bestSlopeRank):
    '''
    Input : 
        f1 : This is the feature on which the tree node would be made.
        y : This is the feature representing the different classes.
    Output : 
        The data after partioning about the bestSlopeRank
    About : 
        This function iterates through all the values in the f1 feature and
        then partitions them around the bestSlopeRank
    '''
    slope = []
    classy = []
    for i in range(len(f1)):
        if f1[i] > bestSlopeRank:
            slope.append(f1[i])
            classy.append(y[i])
    return np.array(slope), np.array(classy)


'''
The logic of this code is simple, the data is read from the disk.
The best rank is calculated and then the corrosponding information gain is 
calculated.
'''
y, f1 = getSortedData()
bestSlopeRank, maxInformationGain = getBestSlopeRank(f1, y)
print("The best rank (slope) to partition is : " + str(bestSlopeRank))
print("The max Information Gain is : " + str(maxInformationGain))
slope, classy = partition(f1, y, bestSlopeRank)

bestSlopeRank, maxInformationGain = getBestSlopeRank(slope, classy)
print("The best rank (slope) to partition is : " + str(bestSlopeRank))
print("The max Information Gain is : " + str(maxInformationGain))