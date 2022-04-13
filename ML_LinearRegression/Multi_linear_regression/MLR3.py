import numpy as np       
import pandas as pd
import matplotlib.pyplot as plt    
import seaborn as sns 
data=pd.read_csv('/Users/nithish/Downloads/50_Startups.csv')
print(data.describe())
print(data.describe().T)
print(data.head(3))
print(data.tail(3))
print(data.corr())
#sns.histplot(data)
#sns.pairplot(data)
#sns.scatterplot(x='R&D Spend',y='Administration',data=data)
#plt.show()
print(data.isna())
print(data.isna().sum().sum())

print(data.dtypes)
print(data.head())
data=data.drop('State',axis=1)
print(data.head(1))

#Assign the values to x and y......
x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,-1:]
print(y)

#split the data for training and testing......
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


#select the model for this problem .....
from sklearn.linear_model import LinearRegression
mul_reg=LinearRegression()
mul_reg.fit(xtrain,ytrain)
ypred=mul_reg.predict(xtest)
print(ypred)
print("Coefficent/Slope:\t",mul_reg.coef_)
print("Constant/Interupt:\t",mul_reg.intercept_)


#print(data.head(1))
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
print("Mean Squared Error:\t",mean_squared_error(ytest,ypred))
print("Root Mean Squared Error:\t",np.sqrt(mean_squared_error(ytest,ypred)))
print("R2 Squared Value:\t",r2_score(ytest,ypred))
print("Variance score :\t:",explained_variance_score(ytest,ypred))


print(data.head(5))


###Unseen Prediction.......
unseen_pred=mul_reg.predict(np.array([[165349.20,136897.80,471784.10]]))
print("Profit:\t",unseen_pred)
