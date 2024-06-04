import sklearn
import pandas as pd
import  numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

automobile_df = pd.read_csv("datasets/cars.csv")

automobile_df= automobile_df.replace("?",np.nan)

automobile_df['MPG'] = automobile_df['MPG'].fillna(automobile_df['MPG'].mean())

automobile_df = automobile_df.dropna()
automobile_df.drop(['Model'],axis=1,inplace=True)
automobile_df.drop(['bore','stroke','compression-ratio'],axis=1,inplace=True)
# print(automobile_df.isnull().sum())
# print(automobile_df.shape)
# print(automobile_df['Year'].str.isnumeric().value_counts())
# print(automobile_df['Year'].loc[automobile_df['Year'].str.isnumeric() == False])
# print(automobile_df.head(5))
# print(automobile_df.dtypes)
# print(automobile_df.dtypes)
# print(automobile_df['Cylinders'].str.isnumeric().value_counts())
# print(automobile_df['Cylinders'].loc[automobile_df['Cylinders'].str.isnumeric() == False])

extr = automobile_df['Year'].str.extract(r'^(\d{4})',expand=False)
automobile_df['Year'] = pd.to_numeric(extr)
automobile_df['Age'] = datetime.datetime.now().year - automobile_df['Year']
automobile_df.drop(['Year'],inplace=True,axis=1)
cylinders = automobile_df['Cylinders'].loc[automobile_df['Cylinders'] != "-"]
cmean = cylinders.astype(int).mean()
automobile_df['Cylinders'] = automobile_df['Cylinders'].replace('-',cmean).astype(int)
automobile_df['Displacement'] = pd.to_numeric(automobile_df['Displacement'],errors='coerce')
automobile_df['Weight'] = pd.to_numeric(automobile_df['Weight'],errors='coerce')
automobile_df['Acceleration'] = pd.to_numeric(automobile_df['Acceleration'],errors='coerce')

automobile_df['Origin'] = np.where(automobile_df['Origin'].str.contains('US'),'US',automobile_df['Origin'])

automobile_df['Origin'] = np.where(automobile_df['Origin'].str.contains('Japan'),'Japan',automobile_df['Origin'])

automobile_df['Origin'] = np.where(automobile_df['Origin'].str.contains('Europe'),'Europe',automobile_df['Origin'])

# print(automobile_df.describe())

# automobile_df.to_csv('datasets/cars_processed.csv')

plt.figure(figsize=(12,8))

##bar graph
# plt.bar(automobile_df['Age'],automobile_df['MPG'])
# plt.xlabel('Age')
# plt.ylabel('Miles per gallon')

##scatter plot
# plt.scatter(automobile_df['Acceleration'],automobile_df['MPG'])
# plt.xlabel('Acceleration')
# plt.ylabel('Miles per gallon')

# plt.scatter(automobile_df['Weight'],automobile_df['MPG'])
# plt.xlabel('Weight')
# plt.ylabel('Miles per gallon')

# automobile_df.plot.scatter(x='Weight',y='Acceleration',c='Horsepower',colormap='viridis',figsize=(12,8))

# plt.bar(automobile_df['Cylinders'],automobile_df['MPG'])
# plt.xlabel('Cylinders')
# plt.ylabel('Miles per gallon')

automobile_df.drop(['Cylinders','Origin'],axis=1,inplace=True)

cars_corr = automobile_df.corr()
fig,ax = plt.subplots(figsize=(12,8))
sns.heatmap(cars_corr,annot=True)
plt.show()