# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:41:11 2025

@author: Mukta
"""

"""
Problem Statement: -

A pharmaceuticals manufacturing company is conducting a study on a new medicine to treat heart diseases. The company has gathered data from its secondary sources and would like you to provide high level analytical insights on the data. Its aim is to segregate patients depending on their age group and other factors given in the data. Perform PCA and clustering algorithms on the dataset and check if the clusters formed before and after PCA are the same and provide a brief report on your model. You can also explore more ways to improve your model. 
1.1 Business Objective
The primary objective is to analyze the dataset to identify patterns and group patients based on their age and other health-related factors. This will help in understanding the effectiveness of the new medicine for treating heart diseases.
1.2 Constraints
The dataset is a snapshot and may not represent the entire population.
The analysis is limited to the features provided in the dataset.
The results may vary based on the choice of clustering algorithms and PCA components.

"""
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\Data_Science\PCA_Assignment\heart disease.csv")

# Display the first few rows of the dataset
print(data.head())

# Data Pre-processing
# Check for missing values
print(data.isnull().sum())

# Feature Engineering: Scaling the data
features = data.drop('target', axis=1)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Exploratory Data Analysis (EDA)
# Summary statistics
print(data.describe())

# Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(data['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='chol', data=data)
plt.title('Cholesterol Levels by Heart Disease Diagnosis')
plt.xlabel('Heart Disease Diagnosis (0 = No, 1 = Yes)')
plt.ylabel('Cholesterol Levels')
plt.show()

# Model Building
# PCA Analysis
pca = PCA()
pca.fit(scaled_features)

# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

# Plotting the explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Choosing the number of components
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.axvline(x=5, color='g', linestyle='--')
plt.show()

# Transforming the data using PCA
pca = PCA(n_components=5)  # Retaining 5 components
pca_features = pca.fit_transform(scaled_features)

# Clustering before PCA
kmeans_before = KMeans(n_clusters=3, random_state=42)
clusters_before = kmeans_before.fit_predict(scaled_features)

# Clustering after PCA
kmeans_after = KMeans(n_clusters=3, random_state=42)
clusters_after = kmeans_after.fit_predict(pca_features)

# Evaluating clustering performance
silhouette_before = silhouette_score(scaled_features, clusters_before)
silhouette_after = silhouette_score(pca_features, clusters_after)

print("Silhouette Score before PCA:", silhouette_before)
print("Silhouette Score after PCA:", silhouette_after)

# Visualizing clusters before PCA
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=clusters_before, cmap='viridis', alpha=0.5)
plt.title('Clusters Before PCA')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Visualizing clusters after PCA
plt.figure(figsize=(10, 6))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters_after, cmap='viridis', alpha=0.5)
plt.title('Clusters After PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
