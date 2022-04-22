import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt 
import seaborn as sns 
from mlxtend.plotting import plot_decision_regions
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
y=data.Outcome
print(y)

##Split the data for training and testing
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=3,test_size=0.3)
print(xtrain.shape)
print(ytrain.shape)
print(ytest.shape)



## Bulid the knn model 
from sklearn.neighbors import KNeighborsClassifier
test_score=[]
train_score=[]
for i in range(3,15):
    knn=KNeighborsClassifier(i)
    knn.fit(xtrain,ytrain)
    train_score.append(knn.score(xtrain,ytrain))
    test_score.append(knn.score(xtest,ytest))
print(train_score)
print(test_score)

max_test_score=max(test_score)
test_score_ind=[i for i,v in enumerate(test_score) if v==max_test_score]
print('max test score {} % and k={}'.format(max_test_score*100,list(map(lambda x:x+1,test_score_ind))))

max_train_score=max(train_score)
train_score_ind=[i for i,v in enumerate(train_score) if v==max_train_score]
print('max of the train score {} % k={}'.format(max_train_score*100,list(map(lambda x:x+1,test_score_ind))))
"""
##Result Visualization
plt.figure(figsize=(12,6))
sns.lineplot(range(3,15),train_score,markers='*',label='Train_score')
sns.lineplot(range(3,15),test_score,markers='+',label='Test_score')
plt.show()
"""


knn=KNeighborsClassifier(12)
knn.fit(xtrain,ytrain)
score=knn.score(xtest,ytest)
print(score)


"""
##Plot Decision Boundary......
value=20000
width=20000
plot_decision_regions(x.values,y.values,clf=knn,legend=2,
                     filler_feature_values={2:value,3:value,4:value,5:value,6:value,7:value},
                     filler_feature_ranges={2:width,3:width,4:width,5:width,6:width,7:width},
                     X_highlight=xtest.values)
plt.title("Knn with diabetes dataset")
plt.show()
"""

##Performance Testing....
from sklearn.metrics import confusion_matrix
ypred=knn.predict(xtest)
print("Confusion Matrix is :",confusion_matrix(ytest,ypred))
crosstab=pd.crosstab(ytest,ypred,rownames=['True'],colnames=['Predected'],margins=True)
print(crosstab)
"""
###Ploting the confusion matrix...
con_mat=confusion_matrix(ytest,ypred)
sns.heatmap(pd.DataFrame(con_mat),annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
"""
##ROC Curve .....
from sklearn.metrics import roc_curve,roc_auc_score
ypred_proba=knn.predict_proba(xtest)[:,1]

roc_auc_score(ytest,ypred_proba)
fpr,tpr,thresholds=roc_curve(ytest,ypred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='knn')
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()

roc_auc_score(ytest,ypred_proba)


from sklearn.model_selection import GridSearchCV
param_grid={'n_neighbors':np.arange(3,100)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)
print('Best Score:'+str(knn_cv.best_score_))
print('Best Parameters:'+str(knn_cv.best_params_))