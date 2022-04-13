
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 
data=pd.read_csv("/Users/nithish/Downloads/50_Startups.csv")
print(data.describe())
print(data.describe().T)
print(data.head())
print(data.isna().sum())
print(data.isna().sum().sum())

#Assigning the values to x & y ..........
x=data.iloc[:,:-1]
x=x.drop('State',axis=1)
print(x)
y=data.iloc[:,-1:]
print(y)


#training and testing......
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.3)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

#select the model.......
from sklearn.linear_model import LinearRegression
base_model=LinearRegression()
base_model.fit(xtrain,ytrain)
ypred=base_model.predict(xtest)
print(ypred)

#predict the unseen values ......
unseen_base=base_model.predict(np.array([[165349.2,136897.8,471784.1]]))
print("Unseen predection for the base model:\t",unseen_base)
print(data.head())
print(x.head(3))




#polynomial model.....
from sklearn.preprocessing import PolynomialFeatures
poly_feat=PolynomialFeatures(degree=3)
x_ploy=poly_feat.fit_transform(x)
print(x_ploy)

#regression for poly model....
from sklearn.linear_model import LinearRegression
poly_model=LinearRegression()
poly_model.fit(x_ploy,y)

#unseen values of the polynomial regression....
unseen_ploy=poly_model.predict(poly_feat.fit_transform(np.array([[165349.2,136897.8,471784.1]])))
print("Poly model predection :\t",unseen_ploy)
print("Unseen predection for the base model:\t",unseen_base)

print(y.head(1))