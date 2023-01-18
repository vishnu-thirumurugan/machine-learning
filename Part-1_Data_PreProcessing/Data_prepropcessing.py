# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:03:49 2023

@author: Vishnu Thirumurugan

Inspired from Hadelin de Ponteves and Krill Eremenko
"""
##################################
######## DATA PRE PROCESSING #####
##################################

####################################
# importing the required libraries #
####################################
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


#########################
# importing the dataset # 
#########################
dataset = pd.read_csv("Data.csv") # creating the dataframe with the in-built function read_csv with file name followed by the extension
# print(dataset)
# create two entities - one to save the independent variable and the other to save the dependent variable 
# The dependent variable is usually located at the end of the data set - the one which we are going to predict with our machine learning models

# data specific comments 
# in this data, we store the country, age and the salery in variable 'X'
# we then store the 'Purchased/ Not purchased' in the variable 'y'

X = dataset.iloc[ : , : -1].values # matrix of features          # all rows and all columns except last one
y = dataset.iloc[ : , -1]. values  # dependent variable vector   # all rows and and last column (label)

# here iloc means locate index. first half represents the rows and the second half represents column.
# range is represented by the ":" sign. 

# let us print the processed data now
print(X)
print(y)


# EXPLANATION FOR CLASS, OBJECT AND METHOD

# A class is the model, or a blueprint, of something we want to build. For example, if we make a house construction plan that gathers the instructions on how to build a house, then this construction plan is the class.

# An object is an instance of the class. So if we take that same example of the house construction plan, then an object is simply a house. A house (the object) that was built by following the instructions of the construction plan (the class).
# And therefore there can be many objects of the same class, because we can build many houses from the construction plan.

# A method is a tool we can use on the object to complete a specific action. So in this same example, a tool can be to open the main door of the house if a guest is coming. A method can also be seen as a function that is applied onto the object, takes some inputs (that were defined in the class) and returns some output.


###############################
# Taking care of missing data # 
###############################

# sckit learn libraray contains most of the data pre processing tools

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') # creating an instance of the class Simple Imputer
# the missing values in the column is replaced by the mean value of that column

# the fit method acts as a bridge between the data and the scikit lib
imputer.fit(X[: , 1 : 3 ]) # we need to go through all the rows and the columns that contains the numerical values. In our case column 2 and 3 contains the numerical data
X[: , 1 : 3 ] = imputer.transform(X[: , 1 : 3 ]) # the transform method returns the empty columns filled with mean values, that is mentioneed in the created object.
print(X)

#################################
# Encoding the categorical data #
#################################

# encoding independent variable 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# create instances of column transformer class
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder= 'passthrough')
# column transformer has 2 arguments, transformers and remainder
# transfomer has 3 inputs - encoder, type of encoder, and the columns that need to be encoded(columns with strings as features)
# column transformer has a method called fit_transform that can be used in a single step to both fit and transform
# it does not return the things in numpy array, So we force it to numpy array as we build our model in numpy array in future.
X = np.array(ct.fit_transform(X))
print(X)

# encoding dependent variable 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # label encoder is a class in sk library. So, we need to include the parenthesis at the end of the class name while ctreating the instances
y  = le.fit_transform(y) 
print(y)


########################################################
# Splitting the dataset into training and testing sets #
########################################################
# when should we want to do feature scaling. before splitting the data or after splitting the data
# feature scaling is done to ensure that all the variables take the values in same scale. This is done to prevent the dominance of one feature over the other.
# feature scaling is done after splitting the data
# this is because, the mean, median or whatever we get on the data should not contain the influence from the test data part
# that is to prevent information leakage from test data.

# the last tool that we will use in the data preprocessing toolkit is the model selection, which contains the function train_test_split
# this function will create 4 sets x test, y test, x train,y train: In both the cases, X will be the matrix of features, y  will be the dependent variable vector.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1 )

# 4 arguments for train_test_split - matrix of features, dependent variable vector, percentage of test data and random splitting to be on 
 
print(X_train)
print(X_test)
print(y_train)
print(y_test)


###################
# Feature Scaling #
###################
# feature scaling should not be applied to dummy variables. dummy variables - encoded categorical values
# Fit will just get the mean and send the deviation of each of your features, and transform will apply this formula to indeed transform your values so that they can all be in the same scale.
from sklearn.preprocessing import StandardScaler 
sc  = StandardScaler()
X_train[: , 3:] = sc.fit_transform(X_train[: , 3:])
X_test[: , 3:] = sc.transform(X_test[: , 3:])

print(X_train)
print(X_test)



 






