#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 17:37:14 2017

@author: virajdeshwal
"""

#lets prepare the dataset by starting the preprocessing of the data.


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

X = X[:, 1:]
#Now lets prepare the training and testing sets

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state  = 0 )

#now lets choose our model and predict the data

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


#building the optimal model using Backward Elimination

#stats model to add extra column of ones for m0.
import statsmodels.formula.api as sm
import numpy as np
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)
#new variable for otimal model

X_opt = X[:, [0,1,2,3,4,5]]
#now instead of linearn regressor model we will use a new model called "Ordinary Least Square".
model_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(model_OLS.summary())

#Now lets remove the x2  because it has highest P value

#type the X_opt again and remoce the x2 index.

X_opt = X[:, [0,1,3,4,5]]
#now instead of linearn regressor model we will use a new model called "Ordinary Least Square".
model_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(model_OLS.summary())

#follow the same procedure untill we did not satisfy the condition P<|t|

#go to step 3 in backward elimination

X_opt = X[:, [0,1,3,4,5]]
#now instead of linearn regressor model we will use a new model called "Ordinary Least Square".
model_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(model_OLS.summary())

#remove x1 this time as it has the highest P value

X_opt = X[:, [0,3,4,5]]
#now instead of linearn regressor model we will use a new model called "Ordinary Least Square".
model_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(model_OLS.summary())

#lets remove the x2 this time . And x2 mean index four in the table. SO, remove 4th index.

X_opt = X[:, [0,3,5]]
#now instead of linearn regressor model we will use a new model called "Ordinary Least Square".
model_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(model_OLS.summary())

#now lets again remove x2 . It mean we have to remove index = 5. As, x1 start at index 4 now.
X_opt = X[:, [0,3]]
#now instead of linearn regressor model we will use a new model called "Ordinary Least Square".
model_OLS = sm.OLS(endog = y, exog= X_opt).fit()
print(model_OLS.summary())

#now we have our optimized model with only one idependent variable.




