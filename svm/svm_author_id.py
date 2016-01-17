#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
clf=SVC(kernel='rbf',C=10000.0)

clf.fit(features_train,labels_train)
labels_pred= clf.predict(features_test)
print labels_pred[10],labels_pred[26],labels_pred[50]
print list(labels_pred).count(1),list(labels_pred).count(0)
accuracy=accuracy_score(labels_test,labels_pred)
print accuracy





#########################################################
### your code goes here ###

#########################################################


