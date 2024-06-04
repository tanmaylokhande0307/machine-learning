import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

advertising_data = pd.read_csv('dataset/Advertising.csv',index_col=0)

plt.figure(figsize=(10,7))

# plt.scatter(advertising_data['newspaper'],advertising_data['sales'])
# plt.xlabel()


advertising_data_corr = advertising_data.corr()
# fig,ax = plt.subplots(figsize=(8,8))
# sns.heatmap(advertising_data_corr,annot=True)
# plt.show()

X = advertising_data['TV'].values.reshape(-1,1)
Y = advertising_data['sales'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.30,random_state=0)

x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)
fit_model = stats_model.fit()
# print(fit_model.summary())

linear_regression = LinearRegression().fit(x_train,y_train)
# print(linear_regression.score(x_train,y_train))

y_pred = linear_regression.predict(x_test)
# print(r2_score(y_test,y_pred))


X_multiple_reg = advertising_data.drop('sales',axis=1)
Y_multiple_reg = advertising_data['sales']

X_multiple_reg.head()