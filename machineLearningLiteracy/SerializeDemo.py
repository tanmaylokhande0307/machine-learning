import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
import pickle

automobiles_data = pd.read_csv('CarPrice_Assignment.csv')

automobiles_data.drop(['car_ID','symboling','CarName'],axis=1,inplace=True)
automobiles_data = pd.get_dummies(automobiles_data)

X = automobiles_data.drop('price',axis=1)
Y = automobiles_data['price']

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

model = LinearRegression().fit(x_train,y_train)
Training_score = model.score(x_train,y_train)

y_pred =  model.predict(x_test)
print(r2_score(y_test,y_pred))

# model_param = {}
# model_param['coef'] = list(model.coef_)
# model_param['intercept'] = model.intercept_.tolist()

# json_txt = json.dumps(model_param,indent=4)

# with open('models/regressor_param.txt','w') as file:
#     file.write(json_txt)
    
# with open('models/regressor_param.txt','r') as file:
#     json_text = json.load(file)   

# json_model = LinearRegression()
# json_model.coef_ = np.array(json_text['coef'])    
# json_model.intercept_ = np.array(json_text['intercept'])

# y_pred1 = json_model.predict(x_test)
# print(r2_score(y_test,y_pred1))

pickle.dump(model,open('models/model.pkl','wb'))
