import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

automobile_df = pd.read_csv('datasets/cars_processed.csv')

# X = automobile_df[['Age']]
# Y = automobile_df['MPG']

# fig,ax = plt.subplots(figsize = (12,8))
# plt.scatter(automobile_df['Age'],automobile_df['MPG'])
# plt.xlabel("Age")
# plt.ylabel("MPG")
# plt.show()

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=7)

# linear_model = LinearRegression().fit(X_train,Y_train)

# print("training score: ",linear_model.score(X_train,Y_train))

# y_pred = linear_model.predict(X_test)
# print(r2_score(Y_test,y_pred))

# X = automobile_df[['Horsepower']]
# Y = automobile_df['MPG']
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=7)

# linear_model = LinearRegression().fit(X_train,Y_train)

# print("training score: ",linear_model.score(X_train,Y_train))

# y_pred = linear_model.predict(X_test)
# print(r2_score(Y_test,y_pred))

# fig,ax = plt.subplots(figsize = (12,8))
# plt.scatter(X_test,Y_test)
# plt.plot(X_test,y_pred,color='r')
# plt.xlabel("Horsepower")
# plt.ylabel("MPG")
# plt.show()

automobile_df = pd.get_dummies(automobile_df,columns=['Origin'])

X = automobile_df.drop('MPG',axis=1)
Y = automobile_df['MPG']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=7)

linear_model = LinearRegression().fit(X_train,Y_train)

print("training score: ",linear_model.score(X_train,Y_train))

y_pred = linear_model.predict(X_test)
print(r2_score(Y_test,y_pred))

