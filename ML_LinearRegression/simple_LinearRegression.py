from cgi import print_directory
import pandas as pd 
import numpy as  np    
import matplotlib.pyplot as plt  
import seaborn as sns   
data = pd.read_csv('/Users/nithish/Downloads/insurance_claims.csv')
print(data)
print(data.isna().sum())
print(data.isna().sum().sum())
print(data.fillna(method='ffill'))
print(data.describe().T)
"""data.hist()
plt.show()
data.boxplot()
plt.show()"""
sns.scatterplot(x='vehicle_claim',y='auto_year',data=data)
plt.show()

