# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:53:23 2019

@author: Monab
"""
import re
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
import xlrd
import openpyxl
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from collections import Counter
#load data
data = pd.read_excel(r'IBM.xlsx', header=None, usecols="C,D", names=['Evidence','Type'])
#print(df.sample(5))
#print(df["Type"])


print(Counter(data["Type"]))

#pre-processing
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\n", "", string)    
    string = re.sub(r"\r", "", string)         #remove all 
    string = re.sub(r"[0-9]", "digit", string) #remove all digits 
    string = re.sub(r"\'", "", string)    #remove all '
    string = re.sub(r"\"", "", string)    #remove all "
    string = re.sub(r"\,", "", string)    #remove all commos
    string = re.sub(r"\]", "", string)    #remove all ]
    string = re.sub(r"\[", "", string)    #remove all [
    string = re.sub(r"REF", "", string) #remove all REF
    return string.strip().lower()
X = []
for i in range(data.shape[0]):
    X.append(clean_str(data.iloc[i][0]))
Y = np.array(data["Type"])

#print(len(Y))
#print(X[3])


#split the data to training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=5)
types = ['EXPERT', 'STUDY', 'ANECDOTAL', 'STUDY, EXPERT', 'EXPERT, ANECDOTAL', 'STUDY, ANECDOTAL', 'STUDY, EXPERT, ANECDOTAL']

print(len(y_test))
###LinearSVC()
unigram_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (1,1), stop_words='english')),('tfidf', TfidfTransformer()),('clf', LinearSVC())])
unigram_clf = unigram_clf.fit(X_train, y_train)

#print(unigram_clf)

#SVM Classification Algorithm
def SVM(Types):
    
    
    #bigram
    #pipeline for text feature extraction and evaluation 
    #tokenizer => transformer => MultinomialNB classifier
    #SGDClassifier(loss='hinge')
    bigram_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (2,2), stop_words='english')),('tfidf', TfidfTransformer()),('clf', LinearSVC())])
    bigram_clf = bigram_clf.fit(X_train, y_train) 
    ##evaluate on test set    
    predicted = bigram_clf.predict(X_test)

    print("******************* NB Classification ********************")
    print("Evidence Types : ", types )
    print("Unigram Accuracy : {}% \n".format(np.mean(predicted == y_test)*100))
    print(metrics.classification_report(y_test, predicted, target_names=types))
    print("Unigram Confusion Matrix : \n", metrics.confusion_matrix(y_test, predicted))
    print("\n")

    print("Bigram Accuracy : {}% \n".format(np.mean(predicted == y_test)*100))
    print(metrics.classification_report(y_test, predicted, target_names= types))
    print("Bigram Confusion Matrix : \n", metrics.confusion_matrix(y_test, predicted))


# NaiveBayes classification algorithm
def NB(Types):
    

    #bigram
    #pipeline for text feature extraction and evaluation 
    #tokenizer => transformer => MultinomialNB classifier
    #SGDClassifier(loss='hinge')
    bigram_clf = Pipeline([('vect', CountVectorizer( analyzer = 'word',ngram_range = (2,2), stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=.01))])
    bigram_clf = bigram_clf.fit(X_train, y_train)
    ##evaluate on test set    
    predicted = bigram_clf.predict(X_test)

    print("******************* NB Classification ********************")
    print("Evidence Types : ", types )
    print("Unigram Accuracy : {}% \n".format(np.mean(predicted == y_test)*100))
    print(metrics.classification_report(y_test, predicted, target_names=types))
    print("Unigram Confusion Matrix : \n", metrics.confusion_matrix(y_test, predicted))
    print("\n")

    print("Bigram Accuracy : {}% \n".format(np.mean(predicted == y_test)*100))
    print(metrics.classification_report(y_test, predicted, target_names= types))
    print("Bigram Confusion Matrix : \n", metrics.confusion_matrix(y_test, predicted))\
    
  
SVM(types)
NB(types)  
    





