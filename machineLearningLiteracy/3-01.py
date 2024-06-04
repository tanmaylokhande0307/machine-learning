import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,KFold

filename = "forestfires.csv"

names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']

df = pd.read_csv(filename,names=names)

pd.set_option("display.max_rows",500)
pd.set_option("display.max_rows",500)
pd.set_option("display.width",1000)

# print(df.shape)
# print(df.dtypes)
# print(df.head())
# print(df.describe()) 

df.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12),inplace=True)
df.day.replace(('sun','mon','tue','wed','thu','fri','sat'),(1,2,3,4,5,6,7),inplace=True)
# print(df.head())
# print(df.corr(method='pearson'))


# df.hist(sharex=False,sharey=False,xlabelsize=15,ylabelsize=15,color='orange',figsize=(15,15))
# plt.suptitle("Histogram",y=1.00,fontweight='bold',fontsize=40)
# plt.show()

# df.plot(kind='density',subplots=True,layout=(7,2),sharex=False,sharey=False,fontsize=16,figsize=(15,15))
# plt.suptitle("pd",y=1.00,fontweight='bold',fontsize=40)
# plt.show()

# plt.figure(figsize=(11,11))
# plt.style.use('default')
# sns.heatmap(df.corr(),annot=True)


