import warnings
import pandas as pd
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score

filename = "forestfires.csv"

names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']

df = pd.read_csv(filename,names=names)
df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
df.day.replace(('sun','mon','tue','wed','thu','fri','sat'),(1,2,3,4,5,6,7),inplace=True)

array = df.values
X = array[:,0:12]
Y = array[:,12]

max_error_scoring = 'max_error'
neg_mean_absolute_error_scoring = 'neg_mean_absolute_error'
r2_scoring = 'r2'
neg_mean_squared_error_scoring = 'neg_mean_squared_error'

models = []
models.append(('LR',LinearRegression()))
models.append(('LASSO',Lasso()))
models.append(('EN',ElasticNet()))
models.append(('Ridge',Ridge()))
models.append(('KNN',KNeighborsRegressor()))
models.append(('CART',DecisionTreeRegressor()))
models.append(('SVR',SVR()))

results = []
names = []

for name,model in models:
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_results1 = cross_val_score(model,X,Y,cv=kfold,scoring=max_error_scoring)
    cv_results2 = cross_val_score(model,X,Y,cv=kfold,scoring=neg_mean_absolute_error_scoring)
    cv_results3 = cross_val_score(model,X,Y,cv=kfold,scoring=r2_scoring)
    cv_results4 = cross_val_score(model,X,Y,cv=kfold,scoring=neg_mean_squared_error_scoring)
    msg = "%s: max error: %f, mean absolute error: %f, r2: %f, mean squared error: %f" % (name,cv_results1.mean(),-cv_results2.mean(),cv_results3.mean(),-cv_results4.mean())
    print(msg)
    


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,shuffle=True,random_state=1)

lassoModel = Lasso()
lassoModel.fit(X_train,Y_train)

predictions = lassoModel.predict(X_test)
print(predictions)