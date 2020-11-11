#Code by Prateek Mishra, IIT2018199 IIIT-Allahabad.
'''
About:
In this code I have implemented a decision tree based on the age feature of the dataset.
This code finds the best rank (age here) using gini impurity to form a decision 
tree of depth 1.

Input:
The input is the heartDisease.csv file.

Output:
Output is the rank (age here) and the gini impurity associated with that rank (age here).

Refrences:
To make this code I took help from the following sources:
1. https://www.youtube.com/watch?v=7VeUPuFGJHk (StatQuest with Josh Starmer)
2. https://drive.google.com/file/d/1s19DQaE7GyGAeYd-HTcAVqfFbux4yRJI/view?usp=sharing (Course Slide)
3. https://www.geeksforgeeks.org/python-pandas-dataframe-sort_values-set-1/ (GeekforGeeks)
'''
import numpy as np
import pandas as pd

def getSortedData():
    '''
    This function reads the data from the dataset file
    and extracts features and sorts them.
    Output : 
        Returns the extracted features.
    '''
    input_data = pd.read_csv("heartDisease.csv")
    # input_data = input_data.sample(frac = 1).reset_index(drop = True) #Shuffling the data.
    input_data.sort_values("f1", axis = 0, ascending = True,
                inplace = True, na_position = 'last')
    y = input_data['y']
    f1 = input_data['f1']
    y = y[0 : 20]
    f1 = f1[0 : 20]
    print(y)
    print(f1)
    return np.array(y), np.array(f1)

def calculateGiniImpurity(lessThanYes, lessThanNo, 
    greaterThanYes, greaterThanNo):
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
        Calculated Gini Impurity.
    About : 
        It calculates the gini impurity.
    '''
    lessThanTotal = lessThanYes + lessThanNo
    lessThanYesProbability = lessThanYes / lessThanTotal
    lessThanNoProbability = lessThanNo / lessThanTotal

    greaterThanTotal = greaterThanYes + greaterThanNo
    greaterThanYesProbability = greaterThanYes / greaterThanTotal
    greaterThanNoProbability = greaterThanNo / greaterThanTotal

    total = lessThanTotal + greaterThanTotal

    lessThanGiniImpurity = 1 - lessThanYesProbability ** 2 - lessThanNoProbability ** 2
    greaterThanGiniImpurity = 1 - greaterThanYesProbability ** 2 - greaterThanNoProbability ** 2
    giniImpurity = ((lessThanTotal / total) * lessThanGiniImpurity) + (greaterThanTotal / total * greaterThanGiniImpurity)
    return giniImpurity

def getGiniImpurity(ageRank, f1, y):
    '''
    Input : 
        ageRank : The rank about which the Gini Impurity is to be calculated
    Output : 
        The Gini Impurity returned by the calculateGiniImpurity helper function.
    About : 
        It calculated the parameters required to calculate the Gini Impurity for 
        the dataset around the ageRank.
    '''
    lessThanYes, lessThanNo, greaterThanYes, greaterThanNo = 0, 0, 0, 0

    for i in range(len(f1)):
        if(f1[i] <= ageRank):
            if(y[i] == 1):
                lessThanYes += 1
            else:
                lessThanNo += 1
        else:
            if (y[i] == 1):
                greaterThanYes += 1
            else:
                greaterThanNo += 1
    giniImpurity = calculateGiniImpurity(lessThanYes, lessThanNo, greaterThanYes, greaterThanNo)
    return giniImpurity

def getbestAgeRank(f1, y):
    '''
    Input : 
        f1 : This is the feature on which the tree node would be made.
    Output : 
        The best rank around which the decision tree node should be made.
    About : 
        This function iterates through all the values in the f1 feature and
        then chooses the one value which has the minimum Gini impurity.
    '''
    uniqueAge = list(set(f1))
    bestAgeRank = f1[0]
    minGiniImpurity = getGiniImpurity(bestAgeRank, f1, y)
    for i in range(len(uniqueAge) - 1):
        ageRank = (uniqueAge[i] + uniqueAge[i + 1]) / 2.0
        temp = getGiniImpurity(ageRank, f1, y)
        print(ageRank, temp)
        if(temp < minGiniImpurity):
            minGiniImpurity = temp
            bestAgeRank = ageRank
    return bestAgeRank

'''
The logic of this code is simple, the data is read from the disk.
The best rank is calculated and then the corrosponding gini impurity is 
calculated.
'''
y, f1 = getSortedData()
bestAgeRank = getbestAgeRank(f1, y)
print("The best rank (age) to partition is : " + str(bestAgeRank))
minGiniImpurity = getGiniImpurity(bestAgeRank, f1, y)
print("The minimum Gini Impurity is : " + str(minGiniImpurity))