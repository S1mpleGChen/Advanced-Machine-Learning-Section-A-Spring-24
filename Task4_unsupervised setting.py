#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:26:22 2024

@author: jeremy chen
"""

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 
  
#using the last two columns's data for the experiment
xvalue = x.iloc[:,2:].values

# Using the elbow method to find the optimal number of clusters
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
wcss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)
    kmeans.fit(xvalue)
    wcss.append(kmeans.inertia_)
# Plot the graph to visualize the Elbow Method to find the optimal number of cluster  
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('Task_4 wcss', dpi=800)
plt.show()


# Applying KMeans to the dataset with the optimal number of cluster

kmeans=KMeans(n_clusters= 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
Y_Kmeans = kmeans.fit_predict(xvalue)

# Visualising the clusters

plt.scatter(xvalue[Y_Kmeans == 0, 0], xvalue[Y_Kmeans == 0,1],s = 10, c='red', label = 'Cluster 1')

plt.scatter(xvalue[Y_Kmeans == 1, 0], xvalue[Y_Kmeans == 1,1],s = 10, c='blue', label = 'Cluster 2')

plt.scatter(xvalue[Y_Kmeans == 2, 0], xvalue[Y_Kmeans == 2,1],s = 10, c='green', label = 'Cluster 3')

plt.scatter(xvalue[Y_Kmeans == 3, 0], xvalue[Y_Kmeans == 3,1],s = 10, c='cyan', label = 'Cluster 4')

plt.scatter(xvalue[Y_Kmeans == 4, 0], xvalue[Y_Kmeans == 4,1],s = 10, c='magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 60, c = 'yellow', label = 'Centroids')
    
plt.title('Clusters of iris')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()
plt.savefig('Task_4 figure', dpi=800)
plt.show()


# for calculate k_mean cost and k_median cost
def compute_kmean(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def compute_kmedian(point, centroid):
    return np.sum(np.abs(point - centroid))


def k_values(X, Y_Kmeans, n_clusters):
    
    k_mean = np.zeros((n_clusters,1))
    k_median = np.zeros((n_clusters,1))
    for i in range(n_clusters):
        for point in X[Y_Kmeans==i]:
            k_mean[i,0]+=(compute_kmean(point, kmeans.cluster_centers_[i,:]))
            k_median[i,0]+=(compute_kmedian(point, kmeans.cluster_centers_[i,:]))
    return k_mean, k_median


k_mean, k_median = k_values(xvalue, Y_Kmeans, n_clusters=4)

print('k_mean value:', np.sum(k_mean))
print('k_median value:', np.sum(k_median))

























