from tkinter import Variable
import pandas as pd    
import numpy as np    
import matplotlib.pyplot as plt   
import seaborn as sns 
data=pd.read_csv('/Users/nithish/Downloads/insurance.csv')
print(data.info())
print(data.isna().sum().sum())
print(data.describe().T)
print(data.head())
##HERE WE HAVE THE STRING VALUES THE ML DOESNT UNDERSTAND THE STRING VALUES
####BY USING LABEL ENCODER WE CAN CHANGE THE STRING--> TO THE NUMERIC...
"""
sns.pairplot(data)
plt.show()
"""

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['sex']=le.fit_transform(data['sex'])

data['smoker']=le.fit_transform(data['smoker'])
data['region']=le.fit_transform(data['region'])
print(data.head())
print(data.describe().T)


####
#data.boxplot()
#data.hist()
"""
sns.scatterplot(x='age',y='charges',data=data)
sns.pairplot(data)
plt.show()
"""

"""
#########Variable Infilation Factor#########
from statsmodels.stats.outliers_influence import variance_infilation_factor
def cal_vif(x):
    vif=pd.DataFrame
    vif['Variables']=data.columns
    vif['vif']=[variance_infilation_factor(data.values,i) for i in range(data.shape[1])]
    return vif
print(cal_vif(data))
"""
#sns.boxplot(data['age'])

#sns.boxplot(data['sex'])
#sns.boxplot(data['bmi'])
#sns.boxplot(data['children'])
#sns.boxplot(data['smoker'])
#sns.boxplot(data['region'])
#sns.boxplot(data['charges'])
#sns.distplot(data['age'])
#sns.distplot(data['bmi'])
#sns.distplot(data['children'])
#sns.distplot(data['charges'])
#plt.show()


print(data.corr())
#here in this we find the least correlation between the independent variables...
#so there is no multicolineartity.....


x=data.iloc[:,:-1]
print(x.head())
y=data.iloc[:,-1:]
print(y.head())


