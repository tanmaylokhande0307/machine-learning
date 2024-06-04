import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

data = pd.read_csv('Mall_Customers.csv', index_col=0)


data.drop('Gender', axis=1, inplace=True)
data.drop('Age', axis=1, inplace=True)

print(data.head())

k_means = KMeans(n_clusters=2)
k_means.fit(data)

print(k_means.labels_)
print(np.unique(k_means.labels_))
centers = k_means.cluster_centers_

print(centers)

# plt.figure(figsize=(10, 8))

# plt.scatter(data['Annual Income (k$)'], 
#             data['Spending Score (1-100)'], 
#             c=k_means.labels_, s=100)

# plt.scatter(centers[:,0], centers[:,1], color='blue', marker='s', s=200) 

# plt.xlabel('Annual Income')
# plt.ylabel('Spending Score')
# plt.title('K-Means with 2 clusters')

# plt.show()


score = silhouette_score (data, k_means.labels_)

print("Score = ", score)

k_means = KMeans(n_clusters=5)
k_means.fit(data)

print(np.unique(k_means.labels_))
centers = k_means.cluster_centers_

plt.figure(figsize=(10, 8))

plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=k_means.labels_, s=100)

plt.scatter(centers[:,0], centers[:,1], color='blue', marker='s', s=200) 

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('5 Cluster K-Means')

plt.show()

score = silhouette_score(data, k_means.labels_)

print("Score = ", score)