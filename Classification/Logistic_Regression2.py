import numpy as np   
import pandas as pd  
import matplotlib.pyplot as plt        
import seaborn as sns 

data=pd.read_csv("/Users/nithish/Downloads/winequality-red.csv")
print(data.head())
#sns.countplot(data['alcohol'])
#plt.show()
print(data.info())
print(data.describe())
print(data.describe().T)
print(data.isna().sum())



##Convert the quality into the categorical form....
data['quality']=data['quality'].astype('category')
print(data.dtypes)




###.........EDA.........
#sns.pairplot(data)
#sns.displot(data)
#sns.distplot(data)
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


###......Assign the values to x&y.....
x=data.iloc[:,:-1]
print(x.head())
y=data.iloc[:,-1:]
print(y.head())



###....Split the data for training and testing......
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.25)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)



#####..........Select the model .......
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(xtrain,ytrain)
ypred=log_reg.predict(xtest)
print(ypred)

### .....Performance Testing.......
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print('Acuuracy score:\t',accuracy_score(ytest,ypred))
print('=========================================================================')
print('Confusion Matrix:\n',confusion_matrix(ytest,ypred))
print('=========================================================================')
print('Classification report:\n',classification_report(ytest,ypred))
print('=========================================================================')


##....ROC curve.....
from sklearn.metrics import roc_auc_score,roc_curve
ypred_prob=log_reg.predict_proba(xtest)[::,1]
fpr,tpr,_=roc_curve(ytest,ypred_prob)
auc=roc_auc_score(ytest,ypred_prob)
plt.plot(fpr,tpr,label='data,auc='+str(auc))
plt.legend(loc=4)
plt.show() 
