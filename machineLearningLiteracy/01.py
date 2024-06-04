import pandas as pd

filename = "forestfires.csv"

names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']

df = pd.read_csv(filename,names=names)

print(df.isnull().sum())