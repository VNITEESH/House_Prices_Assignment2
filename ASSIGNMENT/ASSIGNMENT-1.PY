import pandas as pd  
import numpy as np     
import matplotlib.pyplot as plt     
import seaborn as sns 
data=pd.read_csv('/Users/nithish/Downloads/winequality-red.csv')
print(data.head())
print(data.info())
print(data.tail())
print(data.isna().sum())
print(data.isna().sum().sum())

###....EDA....###
#data.hist()
#sns.pairplot(data)
#sns.countplot(data['quality'])
#plt.show()
"""
plt.subplot(4,3,1)
sns.countplot(x='fixed acidity',hue='quality',data=data)
plt.subplot(4,3,2)
sns.countplot(x='volatile acidity',hue='quality',data=data)

plt.subplot(4,3,3)
sns.countplot(x='citric acid',hue='quality',data=data)

plt.subplot(4,3,4)
sns.countplot(x='residual sugar',hue='quality',data=data)

plt.subplot(4,3,5)
sns.countplot(x='chlorides',hue='quality',data=data)

plt.subplot(4,3,6)
sns.countplot(x='free sulfur dioxide',hue='quality',data=data)

plt.subplot(4,3,7)
sns.countplot(x='total sulfur dioxide',hue='quality',data=data)

plt.subplot(4,3,8)
sns.countplot(x='density',hue='quality',data=data)
plt.subplot(4,3,9)
sns.countplot(x='pH',hue='quality',data=data)

plt.subplot(4,3,10)
sns.countplot(x='sulphates',hue='quality',data=data)
plt.subplot(4,3,11)
sns.countplot(x='alcohol',hue='quality',data=data)
plt.show()

"""

##...Assign the values to x&y...##
x=data.iloc[:,:-1]
print(x.head())
y=data.iloc[:,-1:]
print(y.head())

### split the dat for training and testing....
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)




### Select the base model ....
from sklearn.linear_model import LinearRegression
base_mod=LinearRegression()
base_mod.fit(xtrain,ytrain)
ypred_base=base_mod.predict(xtest)
print(ypred_base)

### ...performance testing.....
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
print("The performance testing is....")
print("=============================================================")
print("Mean squared error:\t",mean_squared_error(ytest,ypred_base))
print("=============================================================")
print("Root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred_base)))
print("=============================================================")
print("R2 Score:\t",r2_score(ytest,ypred_base))
print("=============================================================")



###polynomial based model ....
from sklearn.preprocessing import PolynomialFeatures
poly_feat=PolynomialFeatures(degree=3)
xtrain_ploy=poly_feat.fit_transform(xtrain)
xtest_poly=poly_feat.fit_transform(xtest)
print(xtest_poly)
print(xtrain_ploy)


##apply the model ....
from sklearn.linear_model import LinearRegression
poly_mod=LinearRegression()
poly_mod.fit(xtrain_ploy,ytrain)
ypred_poly=poly_mod.predict(xtest_poly)
print(ypred_poly)


### ...performance testing for the poly model....
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
print("The performance testing is....")
print("=============================================================")
print("Mean squared error:\t",mean_squared_error(ytest,ypred_poly))
print("=============================================================")
print("Root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred_poly)))
print("=============================================================")
print("R2 Score:\t",r2_score(ytest,ypred_poly))
print("=============================================================")


###random forest....
from sklearn.ensemble import RandomForestRegressor
ran_mod=RandomForestRegressor(n_estimators=15)
ran_mod.fit(xtrain,ytrain)
ypred_ran=ran_mod.predict(xtest)
print(ypred_ran)

### ...performance testing for the random forest model....
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
print("The performance testing is....")
print("=============================================================")
print("Mean squared error:\t",mean_squared_error(ytest,ypred_ran))
print("=============================================================")
print("Root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred_ran)))
print("=============================================================")
print("R2 Score:\t",r2_score(ytest,ypred_ran))
print("=============================================================")








### ...performance testing.....
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
print("The performance testing of base model is ....")
#print("=============================================================")
print("Mean squared error:\t",mean_squared_error(ytest,ypred_base))
#print("=============================================================")
print("Root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred_base)))
#print("=============================================================")
print("R2 Score:\t",r2_score(ytest,ypred_base))


print("=============================================================")

### ...performance testing for the poly model....
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
print("The performance testing of poly model is....")
#print("=============================================================")
print("Mean squared error:\t",mean_squared_error(ytest,ypred_poly))
#print("=============================================================")
print("Root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred_poly)))
#print("=============================================================")
print("R2 Score:\t",r2_score(ytest,ypred_poly))



print("=============================================================")

### ...performance testing for the random forest model....
from sklearn.metrics import r2_score,accuracy_score,mean_squared_error
print("The performance testing of random forest is....")
#print("=============================================================")
print("Mean squared error:\t",mean_squared_error(ytest,ypred_ran))
#print("=============================================================")
print("Root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred_ran)))
#print("=============================================================")
print("R2 Score:\t",r2_score(ytest,ypred_ran))
#print("=============================================================")
