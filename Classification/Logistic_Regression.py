from re import I
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt   
import seaborn as sns  

"""
data = pd.read_csv("/Users/nithish/Desktop/diabetes.csv")
print(data.head())
print(data.info())\
print(data.describe())
print(data.describe().T)
print(data.corr())
#data.hist(figsize=(12,10))
#sns.boxplot(data['Glucose'])
#sns.pairplot(data)
#sns.countplot(data['Outcome'])
plt.show()


##In th logistic regression convert the outcomes into catgorical
data['Outcome']=data['Outcome'].astype('category')
print(data.dtypes)

print(data.describe().T)
print(data.isna().isna())
print(data.isna().isna().sum())






#..........Assign the values to x & y....... 
x=data.iloc[:,:-1]
print(x.head(1))
y=data.iloc[:,-1:]
print(y.head(1))

#.......split the data for training and testing.......
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)
print("The Shape Of the Training Sample is:\t",xtrain.shape)
print("The Shape Of the Test Sample is:\t",xtest.shape)
print("The Shape Of the Training Sample is:\t",ytrain.shape)
print("The Shape Of the Test Sample is:\t",ytest.shape)



#.............  Model Selection .............
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(xtrain,ytrain)
#ypred=log.predict(xtest)
#print(ypred)

"""

#######.............RIGHT...................#########


df=pd.read_csv("/Users/nithish/Downloads/diabetes.csv")
print(df.info())
print(df.describe().T)

## ... Convert the outcome into the catogirical form .....
df['Outcome']=df['Outcome'].astype('category')
print(df.dtypes)
print(df.isna().sum())
df.dropna(axis=0,inplace=True)

print(df.isna().sum())


## .. assign the values to x&y....
X=df.iloc[:,:-1]
print(X.head(2))
y=df.iloc[:,-1:]
print(y.head(2))

## split the data for testing and training....
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


##.........model selection..........
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(xtrain,ytrain)
ypred=log.predict(xtest)
print(ypred)

## ....Perforamance testing ........
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print('Performance Testing for this DataSet')
print('=============================================================')
print('Accuracy Score:\t',accuracy_score(ytest,ypred))
print('=============================================================')
print('The confusion Matrix:\n',confusion_matrix(ytest,ypred))
print('=============================================================')
print('Classification Report:\n',classification_report(ytest,ypred))


##......ROC curve ............
from sklearn.metrics import roc_auc_score,roc_curve
ypred_prob=log.predict_proba(xtest)[::,1]
fpr,tpr,_=roc_curve(ytest,ypred_prob)
auc=roc_auc_score(ytest,ypred_prob)
plt.plot(fpr,tpr,label='df,auc='+str(auc))
plt.legend(loc=2)
plt.show()


#unseen_log=log.predict(np.array[[1,5.1,3.5,1.4,0.2]])
#print("flower is :\t",unseen_log)
