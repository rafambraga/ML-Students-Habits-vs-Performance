# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:54:37 2021

@author: rafam
"""

import csv
import pandas as pd
import numpy as np

from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.tree import export_text

filename = 'student-por.csv'

df = pd.read_csv(filename, sep=';', na_values=':', engine='python')
df.head()

df['AG'] = (df.G1 + df.G2 + df.G3) / 3

df.loc[df['AG'] > 10, 'AG_PF'] = 1 #PASS
df.loc[df['AG'] <= 10, 'AG_PF'] = 0 #Fail

df.loc[df['school'] == 'GP', 'school'] = 0 #GP -> 0
df.loc[df['school'] == 'MS', 'school'] = 1 #GP -> 1

df.loc[df['sex'] == 'M', 'sex'] = 0 #M -> 0
df.loc[df['sex'] == 'F', 'sex'] = 1 #F -> 1

df.loc[df['address'] == 'U', 'address'] = 0 #R -> 0
df.loc[df['address'] == 'R', 'address'] = 1 #R -> 1

df.loc[df['famsize'] == 'GT3', 'famsize'] = 0 #GT3 -> 0
df.loc[df['famsize'] == 'LE3', 'famsize'] = 1 #LT3 -> 1

df.loc[df['Pstatus'] == 'T', 'Pstatus'] = 0 #T -> 0
df.loc[df['Pstatus'] == 'A', 'Pstatus'] = 1 #A -> 1

df.loc[df['Pstatus'] == 'T', 'Pstatus'] = 0 #T -> 0
df.loc[df['Pstatus'] == 'A', 'Pstatus'] = 1 #A -> 1

df.loc[df['guardian'] == 'mother', 'guardian'] = 0 #T -> 0
df.loc[df['guardian'] == 'father', 'guardian'] = 1 #A -> 1
df.loc[df['guardian'] == 'other', 'guardian'] = 2 #A -> 1

df.loc[df['schoolsup'] == 'no', 'schoolsup'] = 0 #T -> 0
df.loc[df['schoolsup'] == 'yes', 'schoolsup'] = 1 #A -> 1

df.loc[df['famsup'] == 'no', 'famsup'] = 0 #T -> 0
df.loc[df['famsup'] == 'yes', 'famsup'] = 1 #A -> 1

df.loc[df['paid'] == 'no', 'paid'] = 0 #T -> 0
df.loc[df['paid'] == 'yes', 'paid'] = 1 #A -> 1

df.loc[df['activities'] == 'no', 'activities'] = 0 #T -> 0
df.loc[df['activities'] == 'yes', 'activities'] = 1 #A -> 1

df.loc[df['nursery'] == 'no', 'nursery'] = 0 #T -> 0
df.loc[df['nursery'] == 'yes', 'nursery'] = 1 #A -> 1

df.loc[df['higher'] == 'no', 'higher'] = 0 #T -> 0
df.loc[df['higher'] == 'yes', 'higher'] = 1 #A -> 1

df.loc[df['internet'] == 'no', 'internet'] = 0 #T -> 0
df.loc[df['internet'] == 'yes', 'internet'] = 1 #A -> 1

df.loc[df['romantic'] == 'no', 'romantic'] = 0 #T -> 0
df.loc[df['romantic'] == 'yes', 'romantic'] = 1 #A -> 1

df.loc[df['romantic'] == 'yes', 'romantic'] = 1 #A -> 1

del df['Mjob']
del df['Fjob']
del df['reason']



#print(normalized_df)
print('\n')

bdf = df[['school','sex','address','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','AG_PF']]

bdf.to_csv('bdf.csv', index=False)

training_data = bdf.sample(frac=0.8, random_state=25)
testing_data = bdf.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

training_data.to_csv('bdf_train.csv', index=False)
testing_data.to_csv('bdf_test.csv', index=False)

count = sns.countplot(df['AG_PF']) # Count of how many passes and fails in the set

df = df.drop('G1', axis = 1)
df = df.drop('G2', axis = 1)
df = df.drop('G3', axis = 1)
df = df.drop('AG', axis = 1)

normalized_df=(df-df.min())/(df.max()-df.min())

X  = df.drop('AG_PF', axis = 1)
y = df[['AG_PF']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)

clf_model = DecisionTreeClassifier(criterion="gini", random_state=42,max_depth=5, min_samples_leaf=5)   
clf_model.fit(X_train,y_train)

DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42)

y_predict = clf_model.predict(X_test)

treeAccuracy = accuracy_score(y_test,y_predict)


#create array of probabilities
#y_test_predict_proba = tree.predict_proba(X_test)

# calc confusion matrix
#y_test_predict = tree.predict(X_test[columns])
# print("Confusion Matrix Tree : \n", confusion_matrix(y_test, y_test_predict),"\n")
# print("The precision for Tree is ",precision_score(y_test, y_test_predict)) 
# print("The recall for Tree is ",recall_score(y_test, y_test_predict),"\n") 

print("\nThe decision tree accuracy was ", treeAccuracy, "\n")

target = list(df['AG_PF'].unique())
feature_names = list(X.columns)

#import relevant packages
labels = ['0', '1']

#plt the figure, setting a white background
plt.figure(figsize=(30,10), facecolor ='w')

#create the tree plot
a = tree.plot_tree(clf_model,
                   #use the feature names stored
                   feature_names = feature_names,
                   #use the class names stored
                   class_names = labels,
                   rounded = True,
                   filled = True,
                   fontsize=14)

#show the plot
plt.show()

r = export_text(clf_model, feature_names=feature_names)
print(r)

# Pre processing for KNN
knn_bdf_train = X_train
knn_bdf_train['AG_PF'] =  y_train
knn_bdf_train.to_csv('knn_bdf_train.csv', index=False)

knn_bdf_test = X_test
knn_bdf_test['AG_PF'] =  y_test
knn_bdf_test.to_csv('knn_bdf_test.csv', index=False)

print("1 values: ", df['AG_PF'].value_counts())

