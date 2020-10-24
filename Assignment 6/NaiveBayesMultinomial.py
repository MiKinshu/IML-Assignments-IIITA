import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def getScaledData():
	input_data = pd.read_csv("spam.csv", encoding = "ISO-8859-1", usecols = ["v1", "v2"])
	input_data = input_data.sample(frac = 1).reset_index(drop = True) #shuffling the data.
	input_data['v2'] = input_data['v2'].str.replace(r'[^\w\s]+', '')
	input_data['v2'] = [word.lower() for word in input_data['v2']]

	print(input_data)

	input_data.drop_duplicates(subset = ['v2'], inplace = True) #removing duplicates and resetting index.
	input_data.reset_index(drop = True, inplace = True)

	X_train = []
	X_test = []
	Y_train = []
	Y_test = []

	for i in range(int(0.7 * len(input_data))):
		X_train.append(input_data['v2'][i])
		Y_train.append(input_data['v1'][i])

	for i in range(int(0.7 * len(input_data)), len(input_data)):
		X_test.append(input_data['v2'][i])
		Y_test.append(input_data['v1'][i])
		
	return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

def getDict(X_train, Y_train):
	words = []
	spamWords = []
	hamWords = []
	for i in range(len(X_train)):
		words += (X_train[i].split(" "))
		if Y_train[i] == "spam":
			spamWords += (X_train[i].split(" "))
		else:
			hamWords += (X_train[i].split(" "))
	spamWordSet = set(spamWords)
	hamWordSet = set(hamWords)
	wordset = set(words)
	dictAll = {i:words.count(i) for i in wordset}
	dictSpam = {i:spamWords.count(i) for i in spamWordSet}
	dictHam = {i: hamWords.count(i) for i in hamWordSet}
	return dictAll, dictSpam, dictHam, words, spamWords, hamWords

def probabilityGivenSpam(w, dictSpam, total_spam):
    return (dictSpam[w] / total_spam) #/(total_spam/total_words)

def probabilityGivenHam(w, dictHam, total_ham):
    return (dictHam[w] / total_ham) #/(total_ham/total_words)

def probabilityWord(w, dictAll, total_words):
    return dictAll[w] / total_words

def probabilitySpam(mess, dictSpam, spamWords, dictAll, words):
    num = den = 1
    for w in mess.split():
        if w in spamWords:
            num *= probabilityGivenSpam(w, dictSpam, len(spamWords))
            den *= probabilityWord(w, dictAll, len(words))
    if den == 0:
        num += 1
        den += 1
    num *= len(spamWords) / len(words)
    return num / den

def probabilityHam(mess, dictHam, hamWords, dictAll, words): 
    num = den = 1
    for w in mess.split():
        if w in hamWords:
            num *= probabilityGivenHam(w, dictHam, len(hamWords))
            den *= probabilityWord(w, dictAll, len(words))
    if den == 0:
        num += 1
        den += 1
    num *= len(hamWords) / len(words)
    return num / den

def predict(mess, dictAll, dictSpam, dictHam, words, spamWords, hamWords):
    if probabilitySpam(mess, dictSpam, spamWords, dictAll, words) > probabilityHam(mess, dictHam, hamWords, dictAll, words):
        return "spam"
    else:
        return "ham"

def getaccuracy(X_test, Y_test, dictAll, dictSpam, dictHam, words, spamWords, hamWords):
	corr = 0
	for i in range(len(Y_test)):
		pred = predict(X_test[i], dictAll, dictSpam, dictHam, words, spamWords, hamWords)
		if Y_test[i] == pred:
				corr += 1
	acc = corr / len(Y_test) * 100
	return "Accuracy = " + str(acc)

X_train, X_test, Y_train, Y_test = getScaledData()
dictAll, dictSpam, dictHam, words, spamWords, hamWords = getDict(X_train, Y_train)
print(getaccuracy(X_test, Y_test, dictAll, dictSpam, dictHam, words, spamWords, hamWords))