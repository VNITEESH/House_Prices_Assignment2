import numpy as np         
import pandas as pd  
import matplotlib.pyplot as plt    
import seaborn as sns 
test=pd.read_csv('/Users/nithish/Downloads/test.csv')
print(test)
print(test.info())
print(test.head())
print(test.tail())
"""
sns.scatterplot(x='x',y='y',data=test)
plt.show()

sns.boxplot(x='x',y='y',data=test)
"""
#assigning the values....
a=test.iloc[:,:1]
b=test.iloc[:,1:]
print(a)
print(b)
print(a.head())
print(b.head())
"""
sns.scatterplot(a,b)
plt.show()

plt.scatter(a,b)
plt.show()
"""
##split the data for training and testing.....
from sklearn.model_selection import train_test_split

atrain,atest,btrain,btest=train_test_split(a,b,test_size=0.2,random_state=2)
print(atrain.shape)
print(atest.shape)
print(btrain.shape)



##built the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(atrain,btrain)
bpred=lin_reg.predict(atest)
print(bpred)
print(bpred.shape)
apred=lin_reg.predict(atrain)
#apred=lin_reg.predict(atrain)


#visualizing the Test set results
plt.scatter(atrain,btrain,color='black')
#plt.plot(atrain,apred,color='red')
plt.show()

print('slope / coefficent:\t',lin_reg.coef_)
print()
print('constant / intercept:\t',lin_reg.intercept_)
print()



#performance estimation - cost function r-squared value.....
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
print('MSE:\t',mean_squared_error(btest,bpred))
print()
print('RMSE:\t',np.sqrt(mean_squared_error(btest,bpred)))
print()
print('R2-score:\t',r2_score(btest,bpred))
