# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:29:25 2024

@author: Mukta
"""
'''
1.Problem Statement:
Perform clustering analysis on the telecom dataset.
 The data is a mixture of both categorical and 
 numerical data. It consists of the number of 
 customers who churn. Derive insights and get
 possible information on factors that may affect the
 churn decision. Refer to Telco_customer_churn.xlsx
 dataset.

1.1.Business Objective
The goal of clustering analysis on the telecom dataset is to:

Identify patterns in customer churn by grouping customers based on shared characteristics.
Segment customers into distinct clusters to better understand behavior, risk factors, and key drivers influencing churn.
Develop targeted interventions to reduce churn by tailoring retention strategies for at-risk segments.
Optimize service offerings by identifying customers who might respond to premium or personalized services.
Maximize profitability by improving customer lifetime value through proactive engagement.

1.2.Business Constraints:

1.Timely actions are critical—interventions based on insights must be implemented swiftly to reduce churn.
2.Cost-effectiveness must be considered when designing retention strategies to ensure they align with business goals.

'''
#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

#Load the Dataset
# Load the telecom customer churn dataset
df = pd.read_excel(r"C:\Users\Mukta\OneDrive\Desktop\DS\Dataset\Telco_customer_churn.xlsx")

# Display the first few rows
print(df.head())

# Check for basic info and null values
print(df.info())
print(df.isnull().sum())

#Data Pre-processing
#Handle Missing Values
# Drop rows with missing values (if minimal)
df = df.dropna()

# Alternatively, impute missing values if necessary (e.g., with mode/mean)

#Encode Categorical Variables
# Label encoding for binary categorical variables (like 'Yes'/'No')
binary_cols = ['Churn', 'Partner', 'Dependents']
for col in binary_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# One-Hot Encoding for multi-category variables (like 'InternetService')
df = pd.get_dummies(df, drop_first=True)

print(df.head())  # Verify the encoding

#Scale Numerical Features
# Scale numerical features to ensure equal weighting
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

#Determine Optimal Number of Clusters (Elbow Method & Silhouette Score)
wcss = []  # List to store WCSS values

# Loop to compute WCSS for 1 to 10 clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#  Compute Silhouette Score for additional validation
for n_clusters in range(2, 6):
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    preds = clusterer.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, preds)
    print(f'Silhouette Score for {n_clusters} clusters: {score}')

#Apply K-means Clustering
# Based on the Elbow Method, apply K-means clustering with optimal clusters (e.g., 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# View the cluster assignments
print(df['Cluster'].value_counts())

