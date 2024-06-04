import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10000,random_state=0)

diabetes = pd.read_csv('datasets/diabetes_processed_incomplete.csv')
diabetes_features = diabetes.drop('Outcome',axis=1)
diabetes_label = diabetes[['Outcome']]

imp.fit(diabetes_features)

diabetes_features_array = imp.transform(diabetes_features)

diabetes_features = pd.DataFrame(diabetes_features_array,columns=diabetes_features.columns)

diabetes = pd.concat([diabetes_features,diabetes_label],axis=1)
    