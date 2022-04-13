from matplotlib.transforms import Transform
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import linear_model 
data=pd.read_csv('/Users/nithish/Downloads/insurance.csv')
print(data.head())
print(data.describe().T)
#sns.boxplot(data['age'])
#sns.boxplot(data['bmi'])
#sns.boxplot(data['children'])
#sns.boxplot(data['charges'])
#sns.pairplot(data)

#plt.show()

#Assign the values....
x=data.iloc[:,:-1].values
print(x)

y=data.iloc[:,-1:].values
print(y)

##
from sklearn.compose import ColumnTransformer,make_column_transformer
from sklearn.preprocessing import Normalizer,OneHotEncoder,StandardScaler

colT=ColumnTransformer(remainder='drop',transformers=[("dummy_gender",OneHotEncoder(categories='auto'),[1])])
genders=colT.fit_transform(x)
genders=genders[:,1:]
print(genders)
print(data.corr()) 
#,preprocess the smoker...
colT=ColumnTransformer(remainder='drop',transformers=[("dummy_smoker",OneHotEncoder(categories='auto'),[4])])
smokers=colT.fit_transform(x)
#preprocess the region....
colT=ColumnTransformer(remainder='drop',transformers=[('dummy_region',OneHotEncoder(categories='auto'),[5])])
reegions=colT.fit_transform(x)



#remove orginal categorial/string features...
x=np.delete(x,[1,4,5],axis=1)
print(x)


#concentate the bummy values...
x=np.concatenate((genders,smokers,reegions.toarray(),x),axis=1)
print(x)
print(x.shape)

##split the data 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
##built multilinear regression..
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(xtrain,ytrain)
ypred=lin_reg.predict(xtest)
print(ypred)
print("Coefficent value:\t",lin_reg.coef_)
print("Constant value:\t",lin_reg.intercept_)


from sklearn.metrics import explained_variance_score, r2_score,mean_squared_error,accuracy_score 
print("mean squared errpr:\t",mean_squared_error(ytest,ypred))
print("root mean squared error:\t",np.sqrt(mean_squared_error(ytest,ypred)))
print("R2 score :\t",r2_score(ytest,ypred))
print(data.head(1))
data=data.drop('sex',axis=1)
data=data.drop('smoker',axis=1)
data=data.drop('region',axis=1)
print(data.head(1))

x=data.iloc[:,:-1]
print(x)
y=data.iloc[:,-1:]
print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

from sklearn.linear_model import LinearRegression
mul_reg2=LinearRegression()
mul_reg2.fit(xtrain,ytrain)
ypred=mul_reg2.predict(xtest)
print(ypred)

from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
print("Mean Squared Error :\t",mean_squared_error(ytest,ypred))
print("R2 score:\t",r2_score(ytest,ypred))
print("Root Mean Squared Error:\t",np.sqrt(mean_squared_error(ytest,ypred)))
print("Accuracy Score:\t",explained_variance_score(ytest,ypred))








