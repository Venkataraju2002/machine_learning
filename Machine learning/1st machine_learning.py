# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:09:28 2024

@author: Venkataraju
"""

import pandas as pd
data=pd.read_csv(r"C:\Users\Venkataraju\Downloads\apple_quality.csv")
#print(data.shape)
#print(data.columns)
#print(data.describe()
#print(data.corr)
#print(data.columns)
print(data.loc[data.Quality])
data.corr
data=data.iloc[:-1]
#print(data.isnull().sum())
data['Quality']=data['Quality'].str.replace('good', '1').str.replace('bad', '0').astype(int)
data=data.drop("A_id",axis=1)
print(data.loc['Quality'])
#print(f'Total duplicate rows: {data.duplicated().sum()}')//delete duplicate
#print(data)
#data[data.isnull().any(axis = 1)] check the missing values are present or not
#data.isunique()
#data[target_data].value_counts()
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
#from sklearn.preprocessing import LabelEncoder
#data['Quality'] = data['Quality'].astype(str)
#label_encoder = LabelEncoder()convert the string to int
#data['Quality'] = label_encoder.fit_transform(data['Quality'])
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
mode= RandomForestClassifier()
mode.fit(x_train,y_train)
pred=mode.predict(x_test)
print(accuracy_score(pred,y_test)*100)
print(confusion_matrix(y_test,pred))






