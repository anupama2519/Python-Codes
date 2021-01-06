# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:21:11 2020

@author: admin
"""

import pandas as pd
import numpy as np
import re

#Read the dataset:
data_sp = pd.read_csv('C:/Users/admin/Desktop/python program files/Python Code/data.csv')

data_sp.head()

data_pp= data_sp.iloc[:,2:27]
data_pp.head()

data_pp.replace('[^a-zA-Z]'," ", regex=True,inplace=True)
#Replace all the punctuations

#To lower:
index=data_pp.columns.tolist()
print(index)
for col in index:
   data_pp[col]= data_pp[col].str.lower()

data_pp.head()

#Convert data frame in one paragraph:
    
headlines = []
for row in range(0,len(data_pp)):
    headlines.append(' '.join(str(x) for x in data_pp.iloc[row,0:25]))

print(headlines[0])
print(len(headlines)) #4101

# Remove stop words:
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
ps = PorterStemmer()

corpus=[]
#ls= WordNetLemmatizer()
 
for i in range(len(headlines)):
    review = headlines[i]
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Apply Bag of Words/TFIDF/words2vec algorithm to convert data into vector:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,1))   
#data_vec = cv.fit_transform(corpus)
data_vec = cv.fit_transform(headlines)
data_vec.shape
# Split the data into input and output feature:
x= data_vec
y = data_sp['Label']

# Split the data into train and test:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape
y_train.shape
x_test.shape
y_test.shape

#Random forest classifier:
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=200,criterion='entropy')
model = RF.fit(x_train,y_train)

#Predict the data:
pred = model.predict(x_test)

#Performance metrices:
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
acc = accuracy_score(y_test,pred)    
acc    #0.52
matrix = confusion_matrix(y_test,pred)
matrix    
report = classification_report(y_test,pred)
report
