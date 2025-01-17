# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:46:01 2025

@author: Mukta
"""

'''
Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. After that, perform PCA on the dataset and extract the first 3 principal components and make a new dataset with these 3 principal components as the columns. Now, on this new dataset, perform hierarchical and K-means clustering. Compare the results of clustering on the original dataset and clustering on the principal components dataset (use the scree plot technique to obtain the optimum number of clusters in K-means clustering and check if you’re getting similar results with and without PCA).

business objective:

To identify distinct groups of wines based on their chemical properties using clustering techniques.
constraints:

Data quality, computational resources, and interpretability of results.
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\Data_Science\PCA_Assignment\wine.csv")

#Data Pre-processing
# Check for missing values
print(data.isnull().sum())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('Type', axis=1))  # Exclude target variable

#Exploratory Data Analysis (EDA)
import matplotlib.pyplot as plt
import seaborn as sns

# Summary statistics
print(data.describe())

# Univariate analysis
data.hist(bins=15, figsize=(15, 10))
plt.show()

# Bivariate analysis
sns.pairplot(data, hue='Type')
plt.show()

# Model Building
#K-means Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Determine the optimal number of clusters using the elbow method
inertia = []
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Plotting the elbow method
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Plotting silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# PCA Analysis
from sklearn.decomposition import PCA

# Perform PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(scaled_data)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by each component: {explained_variance}')

#Clustering on PCA Data
# K-means clustering on PCA data
inertia_pca = []
silhouette_scores_pca = []

for k in range(2, 11):
    kmeans_pca = KMeans(n_clusters=k, random_state=42)
    kmeans_pca.fit(pca_data)