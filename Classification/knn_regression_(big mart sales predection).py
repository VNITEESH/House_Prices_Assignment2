
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns 
data=pd.read_csv("/Users/nithish/Downloads/Train.csv")
print(data.head(1))
print(data.dtypes)
print(data.corr())
print(data.describe())
print(data.describe().T)

"""
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(),data=data)
plt.show()
"""
print(data.dtypes)
"""
plt.subplot(3,3,1)
sns.distplot(data['Item_Weight'])
plt.show()
plt.subplot(3,3,2)
sns.distplot(data['Item_Visibility'])
plt.show()
plt.subplot(3,3,3)
sns.distplot(data['Item_MRP'])
plt.show()
plt.subplot(3,3,4)
sns.distplot(data['Outlet_Establishment_Year'])
plt.show()
plt.subplot(3,3,5)
sns.distplot(data['Item_Outlet_Sales'])
plt.show()
plt.title("DISTPLOT OF ALL THE ITEMS")
plt.show()


data.hist()
plt.show()


sns.pairplot(data)
plt.show()
"""

###Checking the null valuess.....
print(data.isna().sum)
print(data.isna().sum())
data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)

data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0],inplace=True)
print(data.isna().sum())


##removing the features..
data.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
print(data.head(1))
print(data.dtypes)

## Dummy values...
data1=pd.get_dummies(data)
print(data1.shape)
print(data.shape)

print(data1.dtypes)




### split the data for training and testing
from sklearn.model_selection import train_test_split
train,test=train_test_split(data1,test_size=0.2)
xtrain=train.drop(['Item_Outlet_Sales'],axis=1)
ytrain=train['Item_Outlet_Sales']
xtest=test.drop(['Item_Outlet_Sales'],axis=1)
ytest=test['Item_Outlet_Sales']
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)



## scaling(inorder to remove the stastical domination)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
xtrain_scalar=scaler.fit_transform(xtrain)
xtrain=pd.DataFrame(xtrain_scalar)
xtest_scalar=scaler.fit_transform(xtest)
xtest=pd.DataFrame(xtest_scalar)
print(xtrain.head())



### Bulid the model
from sklearn import neighbors
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt
rmse=[]   ##root mean squared error
for k in range(20):
    k=k+1
    model=neighbors.KNeighborsRegressor(n_neighbors=k)
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    error=sqrt(mean_squared_error(ytest,pred))
    rmse.append(error)
    print("RMSE value for each k=",k,'is',error)
    print('r2 score is',r2_score(ytest,pred))
 
 
"""
rmse_value=pd.DataFrame(rmse)
print(rmse_value)
print(k)
plt.plot(rmse_value)
plt.show()
"""


test=pd.read_csv('/Users/nithish/Downloads/test.csv')
submission=pd.read_csv('/Users/nithish/Downloads/submission.csv')
print(test.head(1))
print(submission.head(1))

print(test.isna().sum())
test['Item_Weight'].fillna(test['Item_Weight'].mean(),inplace=True)
#test['Outlet_Size'].fillna(test['Outlet_Size'].mean(),inplace=True)
print(test.isna().sum())

test=pd.get_dummies(test)

test_sc=scaler.fit_transform(test)
test=pd.DataFrame(test_sc)
print(test)

predict=model.predict(test)
submission['Item_Outlet_Sales']=predict
submission.to_csv('Submit_file.csv',index=False)
submit_file=pd.read_csv('Submit_file.csv')
print(submit_file.head(1))
