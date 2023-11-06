import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
data = pd.read_csv('Mall_Customers.csv')
print(data.head())
selected_columns = ['Annual_Income ', 'Spending_Score ']
X = data[selected_columns]
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_normalized)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='red', label='Centroids')
plt.title('K-means Clustering of Customer Purchase History')
plt.xlabel('Annual_Income ')
plt.ylabel('Spending_Score')
plt.legend()
plt.show()
data['Cluster'] = cluster_labels
