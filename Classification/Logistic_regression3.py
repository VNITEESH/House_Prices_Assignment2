import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
data=pd.read_csv("/Users/nithish/Downloads/loan-predection.csv")
print(data.head(2))
print(data.info())
print(data.describe())
print(data.describe().T)
print(data.isna().sum())




##change the column names in the easy way ....
data.columns=['loan_id','gender','married','dependents','education','self_employed','applicant_income','co-applicant_income','Loan_amount','loan_amount_term','credit_history','property_area','loan_status']
print(data.head(5))

print(data.describe().T)

data['Loan_amount']=data['Loan_amount'].fillna(data['Loan_amount'].mean())
data['credit_history']=data['credit_history'].fillna(data['credit_history'].median())
print(data.isna().sum())
data.dropna(axis=0,inplace=True)
print(data.isna().sum())


##......EDA.........
#sns.pairplot(data)
#data.hist()
"""
plt.subplot(3,3,1)
sns.countplot(x='gender',hue='loan_status',data=data)
plt.subplot(3,3,2)
sns.countplot(x='married',hue='loan_status',data=data)
plt.subplot(3,3,3)
sns.countplot(x='education',hue='loan_status',data=data)
plt.subplot(3,3,4)
sns.countplot(x='dependents',hue='loan_status',data=data)

plt.subplot(3,3,5)
sns.countplot(x='self_employed',hue='loan_status',data=data)
plt.subplot(3,3,6)
sns.countplot(x='property_area',hue='loan_status',data=data)

plt.show()
"""


#convert the loan_status into the category 
data['loan_status']=data['loan_status'].astype('category')

print(data.info())
##convert the string features into the numeric....
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['loan_id']=le.fit_transform(data['loan_id'])
data['gender']=le.fit_transform(data['gender'])
data['married']=le.fit_transform(data['married'])
data['dependents']=le.fit_transform(data['dependents'])
data['education']=le.fit_transform(data['education'])
data['self_employed']=le.fit_transform(data['self_employed'])
data['property_area']=le.fit_transform(data['property_area'])
print(data.dtypes)



##..... ASSIGN THE VALUES X&Y....
x=data.iloc[:,:-1]
print(x.head())
y=data.iloc[:,-1:]
print(y.head())

##....Split the data for training and testing....
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=2,test_size=0.2)
print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)


##....Select the model.....
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(xtrain,ytrain)
ypred=log_reg.predict(xtest)
print(ypred)






