# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:04:36 2023

@author: Vishnu Thirumurugan
"""

##########################################################
# SIMPLE LINEAR REGREESION - SINGLE INDEPENDENT VARIABLE #
##########################################################

# Regression models are used to predict the continous real values like salary for exampple
# If independent variable contains the time, we are forecasting future values 
# Else we are predicting the present values 



# the first three steps are the same from the data preprocessing template
# 1. importing libraraies 
# 2. importing the dataset 
# 3. Splitting the dataset into training set and test set


###########################
# Importing the libraries #
########################### 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


##########################
# Importing  the dataset # 
##########################
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[: , -1].values


########################################################
# splitting the dataset into training set and test set # 
########################################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)


####################################################################
# Training the simple linear regression model omn the training set #
####################################################################
# this is the simplest machine learning model
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(X_train, y_train) # the arguments must be (1. training set of features, training set of lables)

# The fit() method takes the training data as arguments, which can be one array in the case of unsupervised learning, or two arrays in the case of supervised learning. Note that the model is fitted using X and y , but the object holds no reference to X and y .
# after this fit method, we need to predict the test set result. This is done by predict() method


###################################
# Predicting the test set results # 
###################################
y_pred = regressor.predict(X_test) # this gives the predicted values for test set


########################################
# Visualizing the training set results # 
########################################

# first we plot the x_train and y_train (the real values) as a scatter plot
plt.scatter(X_train, y_train, color = 'red')

# next we plot the x_train and y_pred (predicted values for training set)
# this gives us a regression line, as the 'y' values represent the predicted values 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience - Training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
  


####################################
# Visualizing the test set results # 
####################################

# first we plot the x_test and y_test (the real values) as a scatter plot
plt.scatter(X_test, y_test, color = 'red')

# next we plot the x_train and y_pred (predicted values for training set)
# this gives us a regression line, as the 'y' values represent the predicted values 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Salary vs Experience - Test set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
  
 
 
