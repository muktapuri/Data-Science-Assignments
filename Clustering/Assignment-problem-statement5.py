# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:04:09 2024

@author: Mukta
"""

'''
1.Problem Statement:
    Perform clustering on mixed data. Convert the
    categorical variables to numeric by using 
    dummies or label encoding and perform
    normalization techniques. The dataset has the
    details of customers related to their auto 
    insurance. Refer to Autoinsurance.csv dataset.
    
1.1.Business Objective
The objective of performing clustering on the
 Auto Insurance dataset is to:

1.Group customers into segments based on their
 behaviors, demographics, and claim history.
2.Identify patterns and trends in customer
 behavior to predict future risks and potential
 claims.
3.Develop personalized marketing strategies for
 each cluster to improve retention and satisfaction.
4.Optimize policy pricing by identifying high-risk
 and low-risk customer groups.
5.Design targeted interventions such as 
cross-selling or premium service offerings to
 appropriate segments.

1.2.Constraints
1.Data Quality and Missing Values:
Incomplete or inaccurate data can affect clustering performance. Missing values need to be handled properly.
2.Mixed Data Types:

The dataset contains both categorical and numerical data, requiring careful pre-processing through encoding and scaling.
3.Selection of Optimal Clusters:

Identifying the ideal number of clusters is subjective and may require multiple methods (e.g., Elbow method, Silhouette score) for validation.
4.Interpretability of Clusters:

Clusters must be interpretable for business actions; meaningless or overly complex clusters may limit insights.
5.Scalability:

The clustering model needs to be scalable to handle increasing data volumes over time.
6.Timely Business Actions:

Insights derived from clusters need to be acted upon quickly to provide measurable business value.
7.Cost-Effectiveness:

Implementing strategies based on cluster insights must align with the business budget and available resources.

'''
#Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Dataset
# Load the Auto Insurance dataset
df = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\DS\Dataset\AutoInsurance.csv")

# Display the first few rows
print(df.head())

# Get basic info about the dataset
print(df.info())

#Data Pre-processing
#Handle Missing Values
# Check for missing values
print(df.isnull().sum())

# Drop or impute missing values as needed
df = df.dropna()  # Or use imputation techniques

#Encode Categorical Variables
'''
We'll use Label Encoding for binary variables and
 One-Hot Encoding for multi-category variables.
'''
# Label encode binary columns
binary_cols = ['Gender', 'Married']
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# One-Hot Encoding for multi-category variables
df = pd.get_dummies(df, drop_first=True)

print(df.head())  # Verify the transformation

#Scale Numerical Features
# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Scale the numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(df.head())  # Check scaled data

#Determine the Optimal Number of Clusters
# Elbow Method
wcss = []  # Store WCSS for each cluster count

# Compute WCSS for 1 to 10 clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


#Silhouette Score
for n_clusters in range(2, 6):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    preds = clusterer.fit_predict(df)
    score = silhouette_score(df, preds)
    print(f'Silhouette Score for {n_clusters} clusters: {score}')

#Apply K-means Clustering
# Use the optimal number of clusters (e.g., 4) from the Elbow method
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df)

# Check the distribution of clusters
print(df['Cluster'].value_counts())

#Visualize the Clusters Using PCA
# Apply PCA to reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df.drop('Cluster', axis=1))

# Plot the clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('Customer Clusters - PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

