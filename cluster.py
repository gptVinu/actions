import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load and preprocess data
data = pd.read_csv('C:\\Users\\vinyak\\Downloads\\myData.csv') 
data = data.drop(columns=['Outcome'])
data = data.dropna() #remove rows with null values

# Elbow method for optimal k
wcss = [KMeans(n_clusters=k, random_state=0).fit(data).inertia_ for k in range(1, 10)]
plt.plot(range(1, 10), wcss), plt.title('Elbow Method'), plt.xlabel('Number of clusters'), plt.ylabel('WCSS'), plt.show()

# K-means clustering (k=2)
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
data['kmeans_cluster'] = kmeans.labels_

# PCA for visualization
X_std = StandardScaler().fit_transform(data)
X_pca = PCA(n_components=2).fit_transform(X_std)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_), plt.title('K-means Clustering'), plt.xlabel('PCA 1'), plt.ylabel('PCA 2'), plt.show()

# Silhouette score
print(f"Mean Silhouette for K-Means Clustering: {silhouette_score(data.drop('kmeans_cluster', axis=1), data['kmeans_cluster']):.3f}")

# Hierarchical clustering
dendrogram(linkage(data.drop('kmeans_cluster', axis=1), method='ward')), plt.title('Hierarchical Clustering'), plt.show()