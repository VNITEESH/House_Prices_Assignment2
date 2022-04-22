import numpy as np    
import pandas as pd    
import matplotlib.pyplot as plt   
import seaborn as sns
data=pd.read_csv('/Users/nithish/Downloads/train_data.csv')
print(data.head(5))
print(data.shape)
print(data.isna().sum())
print(data.dtypes)


#### EDA .....
#sns.pairplot(data)
#sns.violinplot(data['word_freq_make'])
#plt.show()
#sns.pairplot(data)
plt.show()

data.drop(['Id'],axis=1,inplace=True)
print(data.head(1))

#sns.countplot(data['ham'])
#plt.show()


### split the data ..
x=data.iloc[:,:-1]
y=data.iloc[:,-1:]
print(x.head())
print(y.head())


##training and testing...
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.3)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

## Gaussian Navie bayes....
from sklearn.naive_bayes import GaussianNB
gauss=GaussianNB()
gauss.fit(xtrain,ytrain)
ypred_gauss=gauss.predict(xtest)
#print(ypred_gauss)
print("the test size is:",len(ypred_gauss))

## performance testing for the Gaussian Navie bayes
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('The performance testing for the Gaussian Navie baye is:')
print('=========================================================')
print('The confusion matrix is:',confusion_matrix(ytest,ypred_gauss))
print('=========================================================')
print('The acuracy score is:',accuracy_score(ytest,ypred_gauss))
print('=========================================================')
print('The classifiaction report:',classification_report(ytest,ypred_gauss))




