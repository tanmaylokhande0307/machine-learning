import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


scikit_learn_version = sklearn.__version__
automobile_train = pd.read_csv('automobiles_file1.csv')

automobile_test = pd.read_csv('automobiles_test.csv')

x_train = automobile_train.drop('price',axis=1)
y_train = automobile_train['price']

x_test = automobile_test.drop('price',axis=1)
y_test = automobile_test['price']

regressor_model = RandomForestRegressor(n_estimators=5, warm_start=True)
rfr_model = regressor_model.fit(x_train,y_train)

training_score = rfr_model.score(x_train,y_train)
print(training_score)

y_pred = rfr_model.predict(x_test)

testing_score = r2_score(y_test,y_pred)
print(testing_score)

rfr_model_param = {}

rfr_model_param['model'] = rfr_model
rfr_model_param['sklearn_version'] = scikit_learn_version
rfr_model_param['r2_score'] = testing_score