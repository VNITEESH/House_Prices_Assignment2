import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt 
import seaborn as sns 
data = pd.read_csv("/Users/nithish/Desktop/diabetes.csv")
print(data)
print(data.head())


###checking the null values ...
print(data.isna().sum())
data.dropna(axis=0,inplace=True)
print(data.isna().sum())


###EDA............
#sns.pairplot(data)
#sns.histplot(data)
#plt.hist(data)
#sns.countplot(data['Outcome'])
#sns.heatmap(data)
plt.show()

print(data.dtypes)

print(data.head())

#### NORMALIZING THE DATA .....
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=pd.DataFrame(sc_x.fit_transform(data.drop(['Outcome'],axis=1),),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
print(x.head())