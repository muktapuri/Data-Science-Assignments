# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:03:33 2024

@author: Mukta
"""

'''
1.Problem Statement:
Analyze the information given in the following
 ‘Insurance Policy dataset’ to  create clusters of
 persons falling in the same type. Refer to
 Insurance Dataset.csv

1.1.Business Objective
The objective of clustering the insurance policy dataset is to:

1.Segment policyholders based on their
 characteristics (e.g., age, premium amount,
 claim history).
2.Identify patterns among customers to tailor
 policies and offers for each segment, improving
 customer satisfaction and retention.
3.Optimize risk management by grouping high-risk
 individuals, enabling the company to better
 allocate resources and manage claims.
4.Develop targeted marketing strategies by 
identifying low-risk or under-served customers who
 can be offered additional insurance products or
 upgrades.
5.This analysis helps the insurance company
 maximize profitability, reduce risk, and enhance
 customer experience by offering more personalized
 products.

1.2. Constraints
Data Quality:

Incomplete or missing data could affect clustering accuracy.
Inconsistent data (e.g., outliers, incorrectly formatted entries) needs to be cleaned.
Scalability:

The clustering model might need regular updates to accommodate new customers and policy changes over time.
Interpretability:

While clusters are formed based on similarities in data, deriving meaningful business insights from these clusters might require additional domain expertise.
Feature Sensitivity:

K-means clustering is sensitive to the scale and range of variables, so feature scaling and outlier removal are critical.
Optimal Cluster Identification:

Selecting the appropriate number of clusters is challenging, as it requires trial and error or techniques like the Elbow Method.
Customer Privacy and Regulations:

The company must ensure that customer data is anonymized and the clustering process complies with data protection laws (e.g., GDPR).
By carefully managing these constraints, the insurance company can effectively utilize clustering to gain insights and provide tailored offerings to their policyholders.

'''
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Load the Dataset
# Load the insurance dataset
df = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\DS\Dataset\Insurance Dataset.csv.xls")

# Display the first few rows
print(df.head())

# Check the shape and basic information
print(df.info())

#Data Pre-processing
#Handle Missing Values
# Check for missing values
print(df.isnull().sum())

# Optionally drop or fill missing values
df = df.dropna()  # Dropping rows with missing values
#Encode Categorical Variables (if any)
#If the dataset contains categorical variables, we need to convert them into numerical values using One-Hot Encoding 
df_encoded = pd.get_dummies(df, drop_first=True)  #  One-Hot Encoding

#Scale the Data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Convert back to DataFrame for easy handling
df_scaled = pd.DataFrame(df_scaled, columns=df_encoded.columns)

#Determine the Optimal Number of Clusters (Elbow Method)
# List to store WCSS (Within-Cluster Sum of Squares)
wcss = []

# Compute WCSS for cluster counts from 1 to 10
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

#K-means Clustering
# Apply K-means with the optimal number of clusters (e.g., 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# View the cluster assignments
print(df['Cluster'].value_counts())

#Visualize the Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='Premium', hue='Cluster', data=df, palette='viridis', legend='full')
plt.title('Clusters of Insurance Policy Holders')
plt.show()
