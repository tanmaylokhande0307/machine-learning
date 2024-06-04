import pandas as pd
from sklearn.preprocessing import scale
from  sklearn.model_selection import train_test_split

automobile = pd.read_csv('../2/datasets/cars_processed.csv')
print(automobile.head())

automobile[['Cylinders']] = scale(automobile[['Cylinders']].astype('float64'))
automobile[['Displacement']] = scale(automobile[['Displacement']].astype('float64'))
automobile[['Horsepower']] = scale(automobile[['Horsepower']].astype('float64'))
automobile[['Weight']] = scale(automobile[['Weight']].astype('float64'))
automobile[['Acceleration']] = scale(automobile[['Acceleration']].astype('float64'))
automobile[['Age']] = scale(automobile[['Age']].astype('float64'))
