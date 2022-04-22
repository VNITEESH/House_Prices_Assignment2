

import numpy as np   
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
data=pd.read_csv("/Users/nithish/Downloads/submission.csv")
print(data.head())
print(data)
print(data.dtypes)

print(data.describe())
print(data.describe().T)
print(data.corr())
print(data.dtypes)

###Finding the missing values ....
print(data.isna())
print(data.isna().sum())
##In the above dataset we don't have any missing values .....





###perform EDA(exploratory data analysis.)
#plt.hist(data)
#sns.histplot(data['SalePrice'],kde=True)
#sns.scatterplot(data['SalePrice'])
"""sns.countplot(data['SalePrice'])
plt.scatter(x='Id',y='SalePrice',data=data)
plt.title('scatter plot')
plt.xlabel('Id')
plt.ylabel('SalePrice')
"""
#sns.pairplot(data)
"""
data.hist()
plt.title('histogram')
plt.xlabel('Id')
plt.ylabel('SalePrice')
"""

#sns.heatmap(data)
"""
sns.violinplot(x='SalePrice' ,data=data)
plt.title('violinplot')

sns.boxplot(data['SalePrice'])
plt.title('boxplot')
plt.show()



sns.boxenplot(data['SalePrice'])
plt.title('boxenplot')
plt.show()
"""

####Pearson Correlation.....
from scipy.stats import pearsonr
corr_1=pearsonr(data['Id'],data['SalePrice'])
print(corr_1)


##split the data .....
x=data.iloc[:,:1]
y=data.iloc[:,1:]
print(x.head())
print(y.head())

###Testing and training....
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

### Model selction...
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(xtrain,ytrain)
ypred=lin_model.predict(xtest)
print(ypred)
print(ypred.shape)

###performance testing ......
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score,explained_variance_score
print("Mean Squared Error is :",mean_squared_error(ytest,ypred))
print("........................................................")
print("Root Mean Squared Error :",np.sqrt(mean_squared_error(ytest,ypred)))
print("........................................................")
print("Expalined Variance Score:",explained_variance_score(ytest,ypred))
print("........................................................")
print("R square value:",r2_score(ytest,ypred))
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


###Unseen predection....
unseen_lin=lin_model.predict(np.array([[1461]]))
unseen_poly=poly_model.predict(poly_feat.fit_transform(np.array([[1461]])))

print(data.head(1))
unseen_lin=lin_model.predict(np.array([[1462]]))
unseen_poly=poly_model.predict(poly_feat.fit_transform(np.array([[1462]])))
print(unseen_lin)
print(unseen_poly)
print(data.head(2))


###From this the polynomial model is giving better predection then the linear model.....