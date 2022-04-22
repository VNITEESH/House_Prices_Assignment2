from cgi import print_arguments
from telnetlib import GA
import pandas as pd
import numpy as np  
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
data=pd.read_csv('/Users/nithish/Downloads/Iris.csv')
print(data.head(1))

### checking the null values ....
print(data.isna().sum())

print(data.corr())
###sns.heatmap(data.corr())  correlation heat map
"""
plt.subplot(1,4,1)
sns.displot(data['SepalLengthCm'])
plt.subplot(1,4,2)
sns.displot(data['SepalWidthCm'])
plt.subplot(1,4,3)
sns.displot(data['PetalLengthCm'])
plt.subplot(1,4,4)
sns.displot(data['PetalWidthCm'])
plt.show()


plt.subplot(1,4,1)
sns.violinplot(data['SepalLengthCm'])
plt.subplot(1,4,2)
sns.violinplot(data['SepalWidthCm'])
plt.subplot(1,4,3)
sns.violinplot(data['PetalLengthCm'])
plt.subplot(1,4,4)
sns.violinplot(data['PetalWidthCm'])
plt.show()

plt.subplot(1,4,1)
sns.distplot(data['SepalLengthCm'])
plt.subplot(1,4,2)
sns.distplot(data['SepalWidthCm'])
plt.subplot(1,4,3)
sns.distplot(data['PetalLengthCm'])
plt.subplot(1,4,4)
sns.distplot(data['PetalWidthCm'])
plt.show()


plt.subplot(1,4,1)
sns.scatterplot(data['SepalLengthCm'])
plt.subplot(1,4,2)
sns.scatterplot(data['SepalWidthCm'])
plt.subplot(1,4,3)
sns.scatterplot(data['PetalLengthCm'])
plt.subplot(1,4,4)
sns.scatterplot(data['PetalWidthCm'])
plt.show()


plt.subplot(1,4,1)
sns.boxenplot(data['SepalLengthCm'])
plt.subplot(1,4,2)
sns.boxenplot(data['SepalWidthCm'])
plt.subplot(1,4,3)
sns.boxenplot(data['PetalLengthCm'])
plt.subplot(1,4,4)
sns.boxenplot(data['PetalWidthCm'])
plt.show()


#sns.pairplot(data)
plt.hist(data)
plt.show()
"""

data.drop(['Id'],axis=1,inplace=True)
print(data.head())


#### split the data....
x=data.iloc[:,:-1]
print(x.head())
y=data.iloc[:,-1:]
print(y.head())

### training and testing 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=3)
print(xtrain.shape)
print(len(xtrain))
print(ytrain.shape)
print(len(ytrain))
print(xtest.shape)
print(len(xtest))
print(ytest.shape)
print(len(ytest))

#### Select the model....
from sklearn.naive_bayes import GaussianNB
gaus=GaussianNB()
gaus.fit(xtrain,ytrain)
ypred_gaus=gaus.predict(xtest)
print('the test size of the test model is',len(ypred_gaus))

#### Performance testing ...
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("The performance testing for the GaussianNB")
print('=========================================================')
print("confusion matrix",confusion_matrix(ytest,ypred_gaus))
print('=========================================================')
print("accuracy score",accuracy_score(ytest,ypred_gaus))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_gaus))
print('=========================================================')

###  Multinomial navie bayes....
from sklearn.naive_bayes import MultinomialNB
multi=MultinomialNB()
multi.fit(xtrain,ytrain)
ypred_multi=multi.predict(xtest)
print('the test size is:',len(ypred_multi))

### performance testing for the multinomial
from sklearn.metrics import classification_report,accuracy_score,multilabel_confusion_matrix,confusion_matrix
print("The performace testing for the multinomial navie bayes")
print('=========================================================')
print('multilabel confusion matrix:',multilabel_confusion_matrix(ytest,ypred_multi))
print('=========================================================')
print('accuracy score :',accuracy_score(ytest,ypred_multi))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_multi))
print('=========================================================')


print(multi.class_count_)
print(multi.class_log_prior_)
print(multi.classes_)
print(multi.feature_log_prob_)



### 
from sklearn.naive_bayes import BernoulliNB
bb=BernoulliNB()
bb.fit(xtrain,ytrain)
ypred_bb=bb.predict(xtest)


### performance testing for bernouli...
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("performance testing for bernouli")
print('=========================================================')
print('confusion matrix :',confusion_matrix(ytest,ypred_bb))
print('=========================================================')
print('accuracy score:',accuracy_score(ytest,ypred_bb))
print('=========================================================')
print('classification report:',classification_report(ytest,ypred_bb))
print('=========================================================')


print('accuracy score:',accuracy_score(ytest,ypred_bb))
print('accuracy score :',accuracy_score(ytest,ypred_multi))
print("accuracy score",accuracy_score(ytest,ypred_gaus))

