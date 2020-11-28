# IML Assignments IIITA

This repository contains my solutions to the assignments of the Introduction to Machine learning course taken at IIIT-Allahabad. Similar assignments are given in the Soft Computing course at IIIT-Allahabad. The repository contains one folder for every assignment. Every folder contains the question along with the dataset and my solution for the same.

### Problem statements:
1. Parse website makemytrip.com or goibibo.com to collect all the listed hotel details.
2. Assignments on linear regression.
    1. Annual Revenue data for a company is given:
        * Draw a least square line fitting the data.
        * What is the expected revenue in 2019
        * Analyze expected error in predictions.
    2. A table shows the final semester marks obtained by 10 students selected at random. Find least square line fitting the above data using
        * X as independent variables (regression of Y on X)
        * Y as independent variable (regression of X on Y)
        * If a student receives a mark 96 in ML, what is her/his expected marks in HUR.
        * If a student receives 95 in HUR. What is  her/his expected marks in ML.
        * After plotting a) and b) what conclusions can you draw?    
    3. Experimental results of pressure P for a given mass of gas corresponding to various values of volume (V) is given:
        * Find the parameters n and c
        * Write the equation connecting P and V.
        * Estimate the value of P when V=100
    4. Given a table containing X and Y. Find the least square parabola which fits the data
        Y = W0 + W1X+ W2X^2
    5. Download the COVID -19 data of India for the month of May, 2020 and design a predictor for the number of deaths on a particular day. Hence, predict the number of deaths on  April 20, 2020 and June 10th , 2020. Verify your prediction with the actual number of deaths and hence calculate the accuracy of prediction.
    6. Download the housing price data set of Windsor City of Canada ( provided on my website link). Design a housing price predictor taking only floor area (plot size), number of bedrooms, and number of bathrooms into considerations. Out of total 546 data , you may take 70% for designing the predictor and 30% for validating the design. The predictor design should be done using the following methods
        * Normal equations  with  and without regularization and compare their performances in terms of % error in prediction. ( only allowed to use NumPy library of Python.no other functions/libraries are allowed )
        * Design Predictor using Batch Gradient Descent Algorithm, Stochastic Gradient Algorithm and mini batch Gradient Descent algorithms (determining minibatch size is your choice- here it could be 10, 20, 30 etc.) with and without feature scaling and compare their performances in terms of % error in prediction.(only allowed to use NumPy library of Python, no other functions/libraries are allowed)
        * Design Predictor using Batch Gradient Descent Algorithm, Stochastic Gradient Algorithm and mini batch Gradient Descent algorithms (determining minibatch size is your choice- here it could be 10, 20, 30 etc.) with and without regularization and compare their performances in terms of % error in prediction.(only allowed to use the NumPy library of Python, no other functions/libraries are allowed)
        * Implement the LWR algorithm on the Housing Price data set with different tau values. Find out the tau value which will provide the best fit predictor and hence compare its results with a) , b) and c) above.
3. Implementing Box-Muller Transformation algorithm in Python (use NumPy library only)
4. Assignments on Logistic Regression
    1. Our task is to build a Logistic Regression based classification model that estimates an applicant's probability of getting admission to an institution based on the scores from those two examinations whose data have been provided here (you may use 70% data for training and 30% for testing).
        * Design a Predictor with two basic features which are given using Batch Gradient Descent Algorithm, Stochastic Gradient Algorithm and mini batch Gradient Descent algorithms (determining minibatch size is your choice- here it could be 10, 20, 30 etc.) with and without feature scaling and compare their performances in terms of % error in prediction.(only allowed to use NumPy library of Python, no other functions/libraries are allowed).  
        * Inject more features from the data set  in the model ( at least 6-9) and repeat (a)
        * Add regularization term and repeat (b). Submit comparative analyses of your results.
    2. After gaining experience of solving problem No 1) Design a classifier using logistic regression on Cleveland Medical data set for heart disease diagnosis. The processed dataset with some 13 features have been given with a label that a patient has a heart disease (1) or not (0). This design should have a professional touch within your ML knowledge. Below the data set brief description has been provided. Also, if you want to have more knowledge about the very interesting data set, please go through the link provided.
5. Assignments on GDA - Given Microchip data with two different quality assurance test results. The third digit tells you whether the microchip has passed the quality assurance test (1 means pass, 0 means fail) or not.
    1. Using raw data set as given, create three more features, and from there develop a GDA model. Thereafter, utilize the same to predict whether a Microchip component will be accepted or rejected. May use 70% data for training and 30 % data for testing.
    2. Using the same data set and features, and same 70% of the data for training and 30% for testing, now use Box-Muller transformation to create a new data set having Gaussian distribution within the range of the given data set and create a new Gaussian Discriminant Analysis (GDA) model. Thereafter, utilize the model to predict where a component will be accepted or rejected using the testing data.
    3. Compare the performance of GDA in both the above cases and write a comparative analysis
report with results.
6. Assignment on Naïve Bayes Classifier - Design a Naïve Bayes classifier for filtering Spam and Ham (Normal) messages. Make a comparative study on the performance of all the three models of Naïve Bayes classifier. The SMS data set, together with a readme file from UCI Machine Learning Repository, has been provided in the folder.
7. Two different assignments were given as Assignment 7.
    1. Assignments on ANN
        * Implement Perceptron training algorithms for AND, OR, NAND and NOR gates.
        * How will you verify your trained algorithms? Justify your solution.
    2. Assignment on Support Vector Machine for predicting Heart disease of a patient using Cleveland heart disease dataset. Use the dataset with the following pre-processing and instructions:
        * Use only two features for simplicity- age ( data in column #1) and trestbps (on admission to the hospital, data in column #4, i.e resting blood pressure in mm/Hg)
        * Modify the last column (# 14) from 1 –heart disease & 0 –no heart disease to Y (i)= {1 and -1}.
        * Apply feature scaling methods to the data of Col# 1 and Col# 4.
        * Use 70% data for training and 30% for testing.
8. Two different Assignments were given as Assignment 8.
    1. Assignment on Back Propagation - Using two inputs and one output X-NOR data, train a Neural Network using Back Propagation Algorithm. Don't use any built in function for Back Propagation. Also explain how you will test the network
    2. Implementation of PCA algorithm for Face recognition - Take 60% data as training set and 40 % data as test set, evaluate your classifier on the following factors:
        * Change the value of k and then, see how it changes the classification accuracy. Plot a graph between accuracy and k value to show the comparative study.
        * Add imposters (who do not belong to the training set) into the test set and then recognize it as the not enrolled person.
9. Assignment on BAM - A Bidirectional Associative Memory(BAM)  is required to store the following  M =4 pairs of patterns:
Set A:  X1 =[1 1 1 1 1 1 ]T,  X2 =[-1 -1 -1 -1 -1 -1 ]T, X3 =[1 -1 -1 1 1 1 ]T, X4 =[1 1 -1 -1 -1 -1 ]T
Set B:  Y1=[1 1 1]T, Y2=[-1 -1 -1]T, Y3=[-1 1 1]T, Y4=[1 -1 1]T
Using BAM algorithm, train a W matrix for BAM which can retrieve all the above mentioned 4 pairs.
Hence test the level of weight corrections of the BAM with examples.
10. Assignment #10 on Self Organizing Neural Network - Consider a Kohonen network with 100 neurons arranged in the form of a two-dimensional lattice with 10 rows and 10 columns. The network is required to classify two-dimensional input vectors such that each neuron in the network should respond only to the input vectors occurring in its region. Train the network with 1500 two-dimensional input vectors generated randomly in a square region in the interval between -1 and +1. Select initial synaptic weights randomly in the same interval  (-1 and +1  ) and take the learning rate parameter α is equal to 0.1. Test the performance of the self organizing neurons using the following
Input vectors:
X1=[0.1  0.8]T,  X2=[0.5  -0.2]T, X3=[-0.8  -0.9]T, X4=[-0.0.6  0.9]T.


### Concepts/Techniques:
1. Web scraping (Data generation).
2. Least square lines
3. Linear regression (GDA, feature scaling, regularisation, feature insertion etc)
4. Box Muller transformation
5. Logistic Regression
6. Gaussian Discriminant Analysis
7. Naive Bayes Classifier (Gaussian discriminant analysis, Multivariate bernoulli event model, Multinomial Event model)
8. ANN
9. Perceptron
10. Support Vector Machine
11. Back Propagation
12. Principal Component Analysis
13. A Bidirectional Associative Memory(BAM)
14. Self Organizing Neural Network
15. Kohonen network

### Getting Started

All of these assignments have been done in python3. To run any of these codes, just install any necessary dependencies and compile it in any python compiler after placing the dataset etc in the same folder.

### Prerequisites
This was done after referring to the lectures by Prof G.C. Nandi. If something seems difficult I strongly recommend going through the [ML lecture series by Prof G.C Nandi.](https://www.youtube.com/playlist?list=PLFWlHcAOSQbrYzwTRTyoOiwjNjyCc_J1i). (This contains most of the lectures that were taken in the course, some others were shared on Google drive.)

### Please See
1. I might have did some jugad (hacky fixes) in some of the codes to increase the accuracy of the respective algorithm. You'll know if you look at the code carefully. I have also pointed out such fixes in most of the places but I might have knowingly or unknowingly left some of these at some place. Please make a note of this. My TA's were not vigilant enough but if you directly use these this might end you up in trouble.
2. The quality of the codes in the first few assignments is really poor. Please bear with it. The codes towards the end are really well documented and commented on.
3. If this repository helped you in any way. Don't forget to :star: the repository. This helps others to reach here faster. Sharing is caring :angel:

## Built with
* [Python3](https://www.python.org/)

## Contributing

I am open to contributions. Contact me on [Facebook](https://www.facebook.com/mishraprateekaries) for any queries.

## Authors

* Prateek Mishra

See also the list of [contributors](https://github.com/MiKinshu/IML-Assignments-IIITA/graphs/contributors) who participated in the project.

## Acknowledgements
* I am thankful to Prof. G. C. Nandi (My ML instructor) for giving me these assignments and teaching me the basics of Machine learning.