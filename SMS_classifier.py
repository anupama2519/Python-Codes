# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:31:27 2020

@author: Anupama
"""

#Load required libraries

import pandas as pd
import re   #regular Expression for removing all the punctuations
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords') #important to download stopwords

#Load the dataset

data = pd.read_csv("C:/Users/admin/Desktop/python program files/Python Code/SMSSpamCollection.csv",sep='\t')
data.head()

#Change the column names
data.columns = ['Label','Messages']

#Clean the dataset

ps = PorterStemmer()
corpus = []


for i in range(len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['Messages'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating Bag of Words vector / Document Term Matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)   # max_fearures means  top words like 2500 means top 2500 words
x = cv.fit_transform(corpus).toarray()

#X is a collection of input features
#We need output variable, which is Label - Ham or SPAM

y = pd.get_dummies(data['Label'])
# We need only one column as a output variable

y_out=y.iloc[:,1].values

#Split the data into train and test:

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y_out,test_size=0.3, random_state=0)

#Naive Bayes Classifier :
    
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
model = NB.fit(x_train,y_train)

pred= model.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score

conf_mat = confusion_matrix(y_test,pred)
print(conf_mat)

acc = accuracy_score(y_test,pred)
acc
