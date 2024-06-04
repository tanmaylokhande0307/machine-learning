import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

gosales = pd.read_csv('../2/datasets/GoSales_Tx_LogisticRegression.csv') 
# print(gosales.shape)

# plt.figure(figsize=(10,7))
# pd.value_counts(gosales['IS_TENT']).plot.bar()
# plt.show()

gender = ['M','F']
labelEncoding = LabelEncoder()
labelEncoding = labelEncoding.fit(gender)

gosales['GENDER'] = labelEncoding.transform(gosales['GENDER'].astype(str))

oneHotEncoding = OneHotEncoder()
oneHotEncoding = oneHotEncoding.fit(gosales['MARITAL_STATUS'].values.reshape(-1,1))

onehotlabels = oneHotEncoding.transform(gosales['MARITAL_STATUS'].values.reshape(-1,1)).toarray()

labels_df = pd.DataFrame()

labels_df['MARITAL_STATUS_Married'] = onehotlabels[:,0]
labels_df['MARITAL_STATUS_Single'] = onehotlabels[:,1]
labels_df['MARITAL_STATUS_Unspecified'] = onehotlabels[:,2]

encoded_df = pd.concat([gosales,labels_df],axis=1)
encoded_df.drop('MARITAL_STATUS',axis=1,inplace=True)

gosales = pd.get_dummies(encoded_df, columns=['PROFESSION'])

gosales = pd.get_dummies(gosales)





