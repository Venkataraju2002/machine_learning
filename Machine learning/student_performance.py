# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 22:43:57 2024

@author: Venkataraju
"""
'''def str_int(string_columns,final_data):
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    for i in string_columns:
        final_data[i]=le.fit_transform(final_data[i])
    print(final_data)
    return final_data'''
import pandas as pd
data1=pd.read_csv(r"C:\Users\Venkataraju\OneDrive\Documents\Machine learning\Datasets\student-mat.csv")
data2=pd.read_csv(r"C:\Users\Venkataraju\OneDrive\Documents\Machine learning\Datasets\student-por.csv")
final_data = pd.concat([data1, data2], ignore_index=True)
#print(final_data["sex"])
#if data1.columns.equals(data2.columns):
    #print("Column names are equal")
#else:
    #print("Column names are not equal")
#print(final_data.info)
#string_columns = final_data.select_dtypes(include=['object'])
#str_int(string_columns,final_data)
#print(final_data)
#enc = OneHotEncoder() 
  
# Passing encoded columns 
  
#enc_data = pd.DataFrame(enc.fit_transform( 
    #data[['Gen_new', 'Rem_new']]).toarray()) 
  
# Merge with main 
#New_df = data.join(enc_data) 
  

#from sklearn.preprocessing import OneHotEncoder
#enc=OneHotEncoder()
#for i in final_data:
    #final_data[i].unique()
    #final_data[i].value_counts()
    #final_data[i] = final_data[i].astype('category')
    #final_data[i] = final_data[i].cat.codes 
    #enc_data = enc.fit_transform(final_data[i])
    #final_data1=final_data.join(enc_data)
#print(final_data["age"])
#two_categorial = ['sex', 'Medu', 'traveltime', 'studytime', 'failures', 'goout', 'Dalc', 'absences']
# define nominal attributes to keep
two_categorial = ['school','sex','address', 'famsize','Pstatus', 'guardian','schoolsup','famsup','paid','activities','nursery','higher', 'internet', 'romantic']
multi_categorial=['Mjob','Fjob','reason']
#for i in two_categorial:
    #print(final_data[i])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

for i in two_categorial:
    le=LabelEncoder()
    
    final_data[i]=le.fit_transform(final_data[i])

final_data['Mjob'] = final_data['Mjob'].astype('category') 
final_data['Fjob'] = final_data['Fjob'].astype('category') 
final_data['reason'] = final_data['reason'].astype('category') 
final_data['Mjob'] = final_data['Mjob'].cat.codes
final_data['Fjob'] = final_data['Fjob'].cat.codes
final_data['reason'] = final_data['reason'].cat.codes
enc=OneHotEncoder()
enc_data = pd.DataFrame(enc.fit_transform( 
final_data[['Mjob','Fjob','reason']]).toarray()) 
print(final_data['Fjob'])
    
    

#final_data["age"]=le.fit_transform(final_data["age"])
#print(final_data.head())