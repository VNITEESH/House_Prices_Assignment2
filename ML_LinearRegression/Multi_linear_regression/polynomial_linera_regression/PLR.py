import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns 
data=pd.read_csv("/Users/nithish/Downloads/Salary_Data.csv")

print(data.describe())
print(data.describe().T)
print(data.head(3))

#assign the values to x&y....
x=data.iloc[:,:-1]
print(x)
y=data.iloc[:,-1:]
print(y)

#apply the base model....
from sklearn.linear_model import LinearRegression
base_model=LinearRegression()
base_model.fit(x,y)
y1=base_model.predict(x)
#plt.scatter(x,y,color='yellow')
#plt.title('Base Linear Regression')
#plt.xlabel('Age')
plt.show()


#apply the polymodel....
from sklearn.preprocessing import PolynomialFeatures
ploy_model=PolynomialFeatures(degree=3)
x_ploy=ploy_model.fit_transform(x)
print(x_ploy)

from sklearn.linear_model import LinearRegression
ploy_reg=LinearRegression()
ploy_reg.fit(x_ploy,y)

print(x_ploy.shape)
print(y.shape)
#plt.subplot(1,2,1)
#plt.scatter(x,y)
plt.show()

#predction....
input=10.5
actual=37731
unseen_base=base_model.predict(np.array([[input]]))
print("Unseen Prediction for the Base Moel:\t",unseen_base)

unseen_poly=ploy_reg.predict(ploy_model.fit_transform(np.array([[input]])))
print("Unseen Value for the Ploynomial model:\t",unseen_poly)
print()
print()
base_error=abs(actual-unseen_base)
ploy_error=abs(actual-unseen_poly)
if (base_error<ploy_error):
    print("base_error is performing best..")
else:
    print("ploy_error is performing best...")
    
print(data.tail(5))