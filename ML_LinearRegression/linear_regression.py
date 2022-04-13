
import pandas as pd   
import numpy as np     
import matplotlib.pyplot as plt    
import seaborn as sns   
data = pd.read_csv("/Users/nithish/Downloads/Salary_Data.csv")
print(data)
print(data.info())
print(data.tail())
print(data.describe())
print(data.describe().T)
"""
plt.scatter(x='YearsExperience',y='Salary',data=data)
plt.show()
sns.scatterplot(x='YearsExperience',y='Salary',data=data)
plt.show()
data.hist()
plt.show()
print(data.corr())
sns.heatmap(data.corr(),annot=True,vmin=-1,vmax=1)
plt.show()
sns.distplot(data['YearsExperience'])
plt.show()
from scipy.stats import skew 
print(skew(data['YearsExperience']))
sns.distplot(data['Salary'])
plt.show()

sns.distplot(data['YearsExperience'])
plt.show()

from scipy.stats import skew  
print(skew(data['YearsExperience'])) 
from scipy.stats import kurtosis
print(kurtosis(data['YearsExperience']))

print(data.isna().sum())

sns.barplot(data['YearsExperience'])

sns.barplot(data['Salary'])
plt.show()

print(data.head())
"""

#Assign x&y in generic way.........
x=data.iloc[:,:1]
print(x.head())

y=data.iloc[:,1:]
print(y.head())
"""
plt.scatter(x,y,color='red')
plt.title('Simple linear regression')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()"""
#split data for training and testing......
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)


#bulit LR model by calling the algorithm from sk learn....
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

print('Salary')
print(ypred)

print('slope / coefficient:\t',model.coef_)
print()
print('constant/intercept:\t',model.intercept_)


plt.scatter(xtrain,ytrain,color='yellow')
plt.show()
"""
#plt.plot(xtrain,model.predict(xtrain))
plt.show()
plt.scatter(xtest,ytest,color='blue')
plt.plot(xtest,lin_reg.predict(xtest),color='red')
plt.show()
"""
#performance estimation-cost function and R-squared value.........
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score

print('MSE:\t',mean_squared_error(ytest,ypred))

print('RMSE:\t',np.sqrt(mean_squared_error(ytest,ypred)))

print('R-Squared:\t',r2_score(ytest,ypred))





#Prediction for unseen input value....
unseen_pred=model.predict(np.array([[10]]))
print('Years experience :\t',unseen_pred)





print(data)
unseen_pred=model.predict(np.array([[10.3]]))
print('Years experience :\t',unseen_pred)


#Normalization......
###normalize the data when we built model again..
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
data_ss=ss.fit_transform(data)
print(data_ss) ##this form we cant undetrstand because it is in the array
#for converting the array into dataframe we did that below.......
df=pd.DataFrame(data_ss,columns=data.columns)
print(df)

#assigning the x&y values...
x=df.iloc[:,:1]
print(x.head())

y=df.iloc[:,1:]
print(y.head())

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=1)
print(xtr.shape)
print(xte.shape)
print(ytr.shape)

from sklearn.linear_model import LinearRegression
ss_reg=LinearRegression()
ss_reg.fit(xtr,ytr)
ypred1=ss_reg.predict(xte)
print(ypred1)

from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
print("MSE:\t",mean_squared_error(yte,ypred1))
print("RMSE:\t",np.sqrt(mean_squared_error(yte,ypred1)))
print("R2_score:\t",r2_score(yte,ypred1))

 #####   Create the deployment object
 
### Pickle file ......

import pickle

with open('deploy','wb') as files:
    pickle.dump(model,files)
with open('deploy','rb') as f:
    lin_object=pickle.load(f)
    
print(lin_object.predict([[10.3]]))
print(data.tail())

#### joblib file.....
import joblib
joblib.dump(model,'model_reg')
deploy_object=joblib.load('model_reg')
print(deploy_object.coef_)
print(deploy_object.predict([[10.3]]))























































































































































































































































































































































































































































































































































































































































































































































