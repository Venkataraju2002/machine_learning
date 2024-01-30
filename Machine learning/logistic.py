# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:48:29 2024

@author: Venkataraju
"""

"""import pandas as pd
data=pd.read_csv(r"C:\Users\Venkataraju\Downloads\apple_quality.csv")
#data  understanding
#print(data.shape)
#print(data.size)
#print(data.head())
#print(data.columns)
#print(data['class'].unique())
#print(data['class'].value_counts())
#print(data.describe())
data.corr()
data = data.iloc[:-1] #delete the last row

#data preprocessing-->handling the missing,vaues and correcting erros
#data.isnull-->use the it checks data wheter missing values present or not if it True misssing values are present or else False

#data.isnull().sum()-->rep it counts null values in each column
data.isnull().sum()
#Dropping the missing values
data.dropna()
data.isnull().sum()
#it replace string to integer
data['Quality'] = data['Quality'].str.replace('good', '1').str.replace('bad', '0').astype(int)
#feature matrix
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
print(y) """
import pandas as pd
data=pd.read_csv(r"C:\Users\Venkataraju\Downloads\apple_quality.csv")
data.shape()

