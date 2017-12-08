#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:22:48 2017

@author: virajdeshwal
"""

import pandas as pd 


file = pd.read_csv('50_Startups.csv')

X = file.iloc[:,:-1].values
y= file.iloc[:,4].values

#lets look for any missing data column

from sklearn.preprocessing import Imputer

imputer  = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])


#Now lets convert the tables into numerical form


from sklearn.preprocessing import LabelEncoder


label = LabelEncoder()

X[:,3] = label.fit_transform(X[:,3])


#Now lets avoid the confusion for the ml algo


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [3])
X = one.fit_transform(X).toarray()
#avoiding the dummy variable trap

#to avoid the dummy variabels

X=X[:,1:]

#now lets split the testing and training sets

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 0)

#lets scale the fearures to be in same range 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.fit_transform(x_test)
y_train = scale.fit_transform(y_train)
y_test = scale.fit_transform(y_test)

#now lets choose the model to fit the data

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

y_hat = model.predict(x_test)

import matplotlib.pyplot as plt

plt.scatter(x_train,y_train, color='cyan')
plt.plot(y_hat,x_train, color='orange')
plt.show()


#Noq lets compute an optimal regression with backword elimination 

import statsmodel.formula.api as sm

#first lets append the row for W0 in the given data set

import numpy as np

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)
 

