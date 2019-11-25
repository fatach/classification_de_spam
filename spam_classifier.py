#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 1 21:17:34 2019

@author: ouedraogo abdoul-fatao
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import string

#NLTK
import nltk
from nltk.corpus import stopwords

#sci-kit learn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             log_loss, precision_recall_curve, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.text import FreqDistVisualizer

#Importing the dataset
print("_________________________DATASET_________________________________")
dataset = pd.read_csv("spam_or_not_spam.csv",encoding='latin-1')
print(dataset.head(10))
print('\n')

#Histogramme of representation of number or spam (value 1) and not spam(value 0)
print(dataset.groupby('label').size())
sns.countplot(dataset['label'],label="Count")
plt.show()

#Selection of variables
X =dataset['email']
y= dataset['label']

#Dealing with missing value
print(dataset.isnull().sum())
dataset['email'].fillna(' ', inplace=True)
print(dataset.isnull().sum())
print('\n')
print(dataset['label'].value_counts())

#Splitting the model into train and test set
print("_________________Train and test_________________- ")
x_train,x_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(x_train.shape)
print(x_test.shape)
print('\n')

print("________________________________-Text preparation___________________")
#Converting our NLP text into vector by using the function countvectorizer
print("_____________________Contvervectorizer___________________________")
con_vec = CountVectorizer(stop_words=stopwords.words('english'))
x_train_count= con_vec.fit_transform(x_train)
#print(x_train_count)

#Token Frequency Distribution
feature =con_vec.get_feature_names()
visualizer =FreqDistVisualizer(features=feature,orient='v')
visualizer.fit(x_train_count)
visualizer.show()

#Compute of word count with the function tfidtransformer
print("---------------------TfdiTransformer------------------------------------------------------")
tfidftransformer = TfidfTransformer()
x_train_tfidf =tfidftransformer.fit_transform(x_train_count)
print(x_train_tfidf.shape)

#Transforming text into a meaningful representation of numbers with of function TfidfVectorize
print("-------------------------TfidfVectorize---------------------------")
vectorizer = TfidfVectorizer()
x_train_tfidf =vectorizer.fit_transform(x_train)
print(x_train_tfidf)
print('\n')

#We used 3 algorithms to test our model: SVM model, Random Forest and Decision Tree
#To use an algorithm  we are going to select one and comment the two other algorithms
#  the pipeline is used to assemble several steps that can
#  be cross-validated together while setting different parameters
print("________________Model Selection________________-")
"""
#DecisionTree model
classifier = DecisionTreeClassifier(criterion ='entropy')
classifier.fit(x_train_tfidf, y_train)
#pipiline
text_clf = Pipeline([('tfidf',TfidfVectorizer()), ('clf',DecisionTreeClassifier())])
text_clf.fit(x_train,y_train)

"""
"""
#Random Forest model
classifier =RandomForestClassifier(n_estimators=100)
classifier.fit(x_train_tfidf,y_train)
#pipiline
text_clf = Pipeline([('tfidf',TfidfVectorizer()), ('clf',RandomForestClassifier())])
text_clf.fit(x_train,y_train)
#print(classifier)

"""
# SVM model
from sklearn.svm import SVC
classifier =SVC(C=1,gamma='auto',cache_size=200, kernel='poly')
classifier.fit(x_train_tfidf,y_train)
print(classifier)
#pipiline
text_clf = Pipeline([('tfidf',TfidfVectorizer()), ('clf',SVC())])
text_clf.fit(x_train,y_train)


#prediction
print('+++++++++++++++++++++++Prediction++++++++++++++++++++++++++++')
y_pred= text_clf.predict(x_test)
print('The predicion accuracy:',y_pred)

#Confusion matrix
print("_______________________Confusion Matrix_______________")
print(confusion_matrix(y_test,y_pred))
#
print ("\n Confusion matrix: \n")
skplt.metrics.plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix",text_fontsize='large')
plt.show()
print('\n')

#Classification report
print("_________________Classification report__________________")
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

print("-----------------Test of the model --------------")
#If the output is 0 the email is not a spam ,else if 1 the email is a spam
test = text_clf.predict([dataset.email[2999]])
if (test==0):
    print("This email is not a spam the result is:",test)
else:
    print("This email is a spam, the result is:",test)
