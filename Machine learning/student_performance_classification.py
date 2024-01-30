# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 14:19:11 2024

@author: Venkataraju
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data1=pd.read_csv(r"C:\Users\Venkataraju\OneDrive\Documents\Machine learning\Datasets\student-mat.csv")
data2=pd.read_csv(r"C:\Users\Venkataraju\OneDrive\Documents\Machine learning\Datasets\student-por.csv")
final_data = pd.concat([data1, data2], ignore_index=True)
#print(final_data["sex"])
final_data['total_grade'] = round((final_data['G1']+final_data['G2']+final_data['G3'])/3)
final_data= final_data.drop(['G1','G2','G3'],axis=1)
final_data['total_grade1'] = 'na'
final_data.loc[(final_data.total_grade >= 15) & (final_data.total_grade <= 20), 'total_grade1'] = 'good' 
final_data.loc[(final_data.total_grade >= 10) & (final_data.total_grade <= 14), 'total_grade1'] = 'fair' 
final_data.loc[(final_data.total_grade >= 0) & (final_data.total_grade <= 9), 'total_grade1'] = 'poor' 
#print(final_data.head(5))
two_categorial = ['school','sex','address', 'famsize','Pstatus', 'guardian','schoolsup','famsup','paid','activities','nursery','higher', 'internet', 'romantic']
multi_categorial=['Mjob','Fjob','reason']

#for i in two_categorial:
    #print(final_data[i])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

for i in two_categorial:
    le=LabelEncoder()
    
    final_data[i]=le.fit_transform(final_data[i])

final_data['Mjob'] = final_data['Mjob'].astype('category') 
#print(final_data['Mjob'])
final_data['Fjob'] = final_data['Fjob'].astype('category') 
final_data['reason'] = final_data['reason'].astype('category') 
final_data['Mjob'] = final_data['Mjob'].cat.codes
#print(final_data['Mjob'])
final_data['Fjob'] = final_data['Fjob'].cat.codes
final_data['reason'] = final_data['reason'].cat.codes
enc=OneHotEncoder()
enc_data = pd.DataFrame(enc.fit_transform( 
final_data[['Mjob','Fjob','reason']]).toarray()) 

print(final_data.shape)
#print(final_data.describe())
#print(final_data.isnull().sum())
#final_data=final_data.corr()
#print(final_data.describe())
x=final_data.iloc[:,:-1]
y=final_data.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test)*100)