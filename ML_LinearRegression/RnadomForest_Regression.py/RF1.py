import pandas as pd   
import numpy as np 
import matplotlib.pyplot as plt   
import seaborn as sns 

data=pd.read_csv("/Users/nithish/Downloads/winequality-red.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.describe().T)
data=data.drop("quality",axis=1)
print(data.head())
print(data.shape)


#Assigning the values to x & y.......
x=data.iloc[:,:-1]
print(x.head())
y=data.iloc[:,-1:]
print(y.head())

##EDA......
#data.hist(figsize=(12,10))
#sns.distplot(data['fixed acidity'])
#sns.boxplot(data['fixed acidity'])
#plt.show()




###Split the data for training and testing.....
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)
print(data.shape)
"""
plt.subplot(3,3,3)
plt.scatter(xtrain['fixed acidity'],ytrain)
plt.scatter(xtrain['volatile acidity'],ytrain)
plt.scatter(xtrain['citric acid'],ytrain)
plt.scatter(xtrain['residual sugar'],ytrain)
plt.scatter(xtrain['chlorides'],ytrain)
plt.scatter(xtrain['free sulfur dioxide'],ytrain)
plt.scatter(xtrain['total sulfur dioxide'],ytrain)
plt.scatter(xtrain['density'],ytrain)
plt.scatter(xtrain['pH'],ytrain)
plt.scatter(xtrain['sulphates'],ytrain)
plt.show()
"""
#sns.pairplot(data)
#plt.show()

# select the base model........

from sklearn.linear_model import LinearRegression
base_model=LinearRegression()
base_model.fit(xtrain,ytrain)
ypred_base=base_model.predict(xtest)
print(ypred_base)

"""
plt.scatter(xtrain['fixed acidity'],ytrain)
plt.plot(xtrain['fixed acidity'],base_model.predict(xtrain))
plt.show()
"""
##performance for the base model...
from sklearn.metrics import r2_score,mean_squared_error,explained_variance_score
print("The Mean Squared Error is :\t",mean_squared_error(ytest,ypred_base))
print("........................................................")
print("The Root Mean Squared Error is :\t",np.sqrt(mean_squared_error(ytest,ypred_base)))
print("........................................................")
print("The R2 score is :\t",r2_score(ytest,ypred_base))
print("........................................................")
print("The expected Variance SCore is:\t",explained_variance_score(ytest,ypred_base))
print("........................................................")



## The Polynomial Model.......
from sklearn.preprocessing import PolynomialFeatures
poly_feat=PolynomialFeatures(degree=3)
xtrain_poly=poly_feat.fit_transform(xtrain)
xtest_poly=poly_feat.fit_transform(xtest)
print(xtest_poly)
print(xtrain_poly)

print("........................................................")

## model for the ploynomail ......
from sklearn.linear_model import LinearRegression
poly_model=LinearRegression()
poly_model.fit(xtrain_poly,ytrain)
ypred_poly=poly_model.predict(xtest_poly)
print(ypred_poly)


##performance for the poly model...
from sklearn.metrics import r2_score,mean_squared_error,explained_variance_score
print("The Mean Squared Error is :\t",mean_squared_error(ytest,ypred_poly))
print("........................................................")
print("The Root Mean Squared Error is :\t",np.sqrt(mean_squared_error(ytest,ypred_poly)))
print("........................................................")
print("The R2 score is :\t",r2_score(ytest,ypred_poly))
print("........................................................")
print("The expected Variance SCore is:\t",explained_variance_score(ytest,ypred_poly))
print("........................................................")





## Random Forest Model.......
from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(n_estimators=10)
rf_model.fit(xtrain,ytrain)
ypred_rf=rf_model.predict(xtest)

##performance for the base model...
from sklearn.metrics import r2_score,mean_squared_error,explained_variance_score
print("The Mean Squared Error is :\t",mean_squared_error(ytest,ypred_rf))
print("........................................................")
print("The Root Mean Squared Error is :\t",np.sqrt(mean_squared_error(ytest,ypred_rf)))
print("........................................................")
print("The R2 score is :\t",r2_score(ytest,ypred_rf))
print("........................................................")
print("The expected Variance SCore is:\t",explained_variance_score(ytest,ypred_rf))
print("........................................................")

print(x.head())





#unseen values .....
unseen_base=base_model.predict(np.array([[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56]]))
print("Unseen values Predecting using the base model:\t",unseen_base)
print("........................................................")
unseen_poly=poly_model.predict(poly_feat.fit_transform(np.array([[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56]])))
print("Unseen values Predecting using the poly model:\t",unseen_poly)
print("........................................................")
unseen_rf=rf_model.predict(np.array([[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56]]))
print("Unseen values Predecting using the random forest model:\t",unseen_rf)
print("........................................................")
print(data.head(1))
