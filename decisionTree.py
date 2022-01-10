#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:18:50 2021

@author: cassieschatz
"""

#Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import tree
from matplotlib import pyplot as plt
#%matplotlib inline
print()

print("Getting data...")
#Read csv
dataset = pd.read_csv('Pokemon.csv');

print("Cleaning data...")


#Preparing the data:
Z = dataset['Type 1']
W = dataset['Type 2']


#Get the list of different values:
typeChanger = []
typeChanger.append(Z[0])
i = 0;
for z in Z:
    i = 0;
    
    for tc in typeChanger:
        if (z == tc):
            i = 1
    if(i == 0):
        typeChanger.append(z)


wNew = []
for w in W:
    if(type(w) is float):
        wNew.append("BLANK")
    else:
        wNew.append(w)



typeChanger.append("BLANK")        
le = preprocessing.LabelEncoder()
le.fit(typeChanger)
Z = le.transform(Z) 
W = le.transform(wNew) 



X = dataset.drop(['#','Name', 'Type 1', 'Type 2', 'Legendary'], axis=1)
X.insert(1, "Type 1", Z)
X.insert(2, "Type 2", W)

y = dataset['Legendary']


print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print("Classifying data...")
#Classify Data
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

print("Predicting data...")
#Predict data
y_pred = classifier.predict(X_test)

print("Confuson Matrix, Decision Tree:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

imp = classifier.feature_importances_
plt.bar([x for x in range(len(imp))], imp)
plt.title("Decision Tree's Feature Importance")
plt.show
plt.close()

print("Confusion Tree, Random Forest: ")
#Using the random forest
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Feature Importance
impRF = clf.feature_importances_
plt.bar([x for x in range(len(impRF))], impRF)
plt.title("Random Forest's Feature Importance")
plt.show



