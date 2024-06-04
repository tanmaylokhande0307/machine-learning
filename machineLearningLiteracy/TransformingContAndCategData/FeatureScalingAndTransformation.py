import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler,Normalizer, Binarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


np.set_printoptions(precision=3)

diabetes = pd.read_csv("../2/datasets/diabetes_processed.csv")

features_df = diabetes.drop('Outcome',axis=1)
target_df = diabetes['Outcome']
print(target_df)

scaler = MinMaxScaler(feature_range=(0,1))
rescaled_features = scaler.fit_transform(features_df)

rescaled_features_df = pd.DataFrame(rescaled_features,columns=features_df.columns)

standard_scaler = StandardScaler()

standard_scaler = standard_scaler.fit(features_df)
standardized_features = standard_scaler.transform(features_df)

standardized_features_df = pd.DataFrame(standardized_features,columns=features_df.columns)

# standardized_features_df.boxplot(figsize=(15,7),rot=45)
# plt.show()  

normalizer = Normalizer(norm='l1')
normalized_features = normalizer.fit_transform(features_df)
normalized_features_df = pd.DataFrame(normalized_features,columns=features_df.columns)

l2normalizer = Normalizer(norm='l2')
l2normalized_features = l2normalizer.fit_transform(features_df)
l2normalized_features_df = pd.DataFrame(l2normalized_features,columns=features_df.columns)

maxNormalizer = Normalizer('max')
maxnormalized_features = maxNormalizer.fit_transform(features_df)
maxnormalized_features_df = pd.DataFrame(maxnormalized_features,columns=features_df.columns)

binarizer = Binarizer(threshold=float((features_df[['Pregnancies']].iloc[0]).mean()))
binarized_features = binarizer.fit_transform(features_df[['Pregnancies']])

for i in range(1,features_df.shape[1]):
    scaler = Binarizer(threshold=float((features_df[[features_df.columns[i]]].iloc[0]).mean())).fit(features_df[[features_df.columns[i]]])
    new_binarized_feature =  scaler.transform(features_df[[features_df.columns[i]]])
    binarized_features = np.concatenate((binarized_features,new_binarized_feature),axis=1)
    

def build_model(X,Y,test_fraction):
    x_train,y_train,x_test,y_test = train_test_split(X,Y,test_size=test_fraction)
    model = LogisticRegression(solver='liblinear').fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Test score: ",accuracy_score(y_test,y_pred))    
 
build_model(rescaled_features,target_df,0.2)    