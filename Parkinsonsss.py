# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:24:17 2019

@author: Shivam Bhargava
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import math  


df=pd.read_csv('file:///C:/Users/Shivam Bhargava/Desktop/Parkinsons/parkinsonsdisease/Data1.csv')
print(df.describe)
x=df.iloc[:,0:18].values
y=df.iloc[:,19].values
print("\nPrinting Shape of X :\n",x.shape)
print("\nPrinting Shape of Y :\n",y.shape)
print("\nTotal No of Unique Values :\n ",np.unique(y).sum())
print("\nPrinting Unique Values of Y :\n",np.unique(y))
print("\nNumber of Attritube :\n",x.shape[1])
print("\nNumber of Instance :\n",x.shape[0])


#Missing Values
print("\nChecking Null Values in our Parkinsons Dataset :\n")
print(df.isnull().sum())
print("\nAs we can see there is no Null Values in our Datasets\n")
print("Lets Suppose if we had Null Values,\nThen we can either use 'Drop' or 'Imputer Method' to correct it\n")

'''
df=pd.read_csv(StringIO(csv_data))
dl=df.dropna() ##it will drop the values which wil contain null values
print("after dropping\n",dl)
from sklearn.preprocessing import Imputer
imr=Imputer(missing_values='NaN',strategy='mean',axis=0) #it will calculate the avg of no in rows wise and replace the nan value with that no
imr=imr.fit(df)
imputed_data=imr.transform(df.values)
print(imputed_data)
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print("\nScaling the features : ")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

print("\nBefore Applying Training and Testing Split,\nShape of x was :\n",np.shape(x))
print("\nAfter Applying Training and Testing Split,\nShape of x is :")
print("Training Data :",np.shape(x_train))
print("Testing Data :",np.shape(x_test))

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

ch='y'
while(ch=='y'):
    print("\nChoose Which Model You Want To Apply :")
    s= int(input("1. Linear Regression \n2. DecisionTreeRegressor \n3. SVR \n4.KNeighborsRegressor"))
    if s==1:
        error = 1000
        ind = 0
        l=['l1','l2','l3']
        for i in l:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error
            reg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
            reg.fit(x_train,y_train)
            y_pred=reg.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            if rmse < error:
                error = rmse
                ind = i
        print("\nAccuracy Of LinearRegressor :")
        print('Misclassified Samples: %d'%(y_test!=y_pred).sum())
        print(100 - error, "Accuracy With Best Value Of Estimator Is ", ind)
        print("\n\nPress y to continue and n to quit\n")
        ch=(input())
    

    if s==2:
        error=1000
        ind=0
        rmse_val = [] 
        #l=["F1","F2","F3"]
        for i in range(10):
            i=i+1
            from sklearn.tree import DecisionTreeRegressor
            tree = DecisionTreeRegressor(max_depth=i,random_state=0)
            tree.fit(x_train,y_train)
            y_pred=tree.predict(x_test)
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            print('Accuracy Of DecisionTreeRegressor for Max Depth' , i , 'is:', 100-rmse)
            if rmse < error:
                error = rmse
                ind = i
        print("\nAccuracy Of Decision Tree Regressor :")
        print('Misclassified Samples: %d'%(y_test!=y_pred).sum())
        print(100 - rmse, " Accuracy With Best Value Of Estimator Is ", ind)
        print("\n\nPress y to continue and n to quit\n")
        ch=(input())
    

    if s==3:
        error=1000
        ind = 0
        l = ['linear','rbf']
        for i in l:
            from sklearn.svm import SVR
            classifier = SVR()
            classifier.fit(x_train,y_train)
            y_pred = classifier.predict(x_test)    
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            print('Accuracy Of SVR for ' , i , 'is:', 100-rmse)
            if rmse < error:
                error = rmse
                ind = i
            #print best value of estimator
        print("\nAccuracy Of SVR Regressor :")
        print('Misclassified Samples: %d'%(y_test!=y_pred).sum())
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        print("\n\nPress y to continue and n to quit\n")
        ch=(input())
       
    
    
    if s==4:
        rmse_val = [] #to store rmse values for different k
        from sklearn.neighbors import KNeighborsRegressor
        for K in range(5):
            K = K+1
            model = KNeighborsRegressor(n_neighbors = K)
            model.fit(x_train, y_train)  #fit the model
            pred=model.predict(x_test) #make prediction on test set
            from sklearn.metrics import mean_squared_error
            error =np.sqrt(mean_squared_error(y_test,pred)) #calculate rmse
            rmse_val.append(error) #store rmse values
            print('Accuracy Of KNeighborsRegressor for k= ' , K , 'is:', 100-rmse)
            if rmse < error:
                error = rmse
                ind = K
        print("\nAccuracy Of KNeighborsRegressor :")
        print('Misclassified Samples: %d'%(y_test!=y_pred).sum())
        print(100 - error, " Accuracy With Best Value Of Estimator Is ", ind)
        curve = pd.DataFrame(rmse_val) #elbow curve 
        curve.plot()
        print("\n\nPress y to continue and n to quit\n")
        ch=(input())
        
from sklearn.decomposition import PCA       
pca = PCA(n_components=3)
train_pca = pca.fit_transform(x_train)
print('\nRepresentation of dataset in 3 dimensions:\n')
print(train_pca)

for i in range(0,19):
    df.loc[i].plot()