# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:04:45 2024

@author: Mukta
"""

'''
1.Problem Statement:
Perform clustering for the crime data and identify
 the number of clusters  formed and draw inferences. 
 Refer to crime_data.csv dataset.
 
1.1.What is the Business Objective?
Business Objective:
The objective is to perform clustering analysis on
 crime data to identify patterns and group cities
 with similar crime rates. This will help in 
 understanding regional crime patterns, enabling
 authorities and policymakers to design targeted
 strategies for crime prevention and resource
 allocation. The insights derived from clustering
 can aid in identifying:

     1.High-risk cities requiring focused law 
     enforcement efforts.
     2.Moderate-risk cities that may need proactive
     measures to prevent crime escalation.
     3.Low-risk cities that could serve as models
     for crime prevention programs.
The business goal is to optimize public safety 
efforts by identifying cities with similar crime
 profiles and tailoring intervention strategies
 accordingly.

1.2.Constraints:
1.Data Quality:
Clustering results depend heavily on the quality
 and completeness of the data.Missing values or
 outliers could affect clustering performance.
2.Feature Scaling Requirement: 
    Crime data features (like murder, assault, and
    theft rates) have different units and scales,
    requiring standardization for clustering
    algorithms to perform correctly.
3.Cluster Interpretability:
    The clusters should be easily interpretable so
    that authorities can act on the findings.
3.Optimal Cluster Selection:
    Determining the correct number of clusters
    can be challenging and subjective.
4.Static Data: 
    Crime data is often historical and may not
    reflect real-time crime patterns, which can
    limit the immediate applicability of insights.
5.Balancing Resources:
    Identifying high-risk areas must translate into
    actionable strategies, but resource allocation
    across regions can be constrained by budget or
    manpower limitations.
    

'''
#Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Load the Dataset
# Load the crime data
df = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\DS\Dataset\crime_data.csv.xls")

# Check the first few rows of the dataset
print(df.head())

#Data Pre-processing
# Check for missing values
print(df.isnull().sum())

# If missing values are found, handle them appropriately (e.g., fill with mean or drop rows)
df = df.dropna()  # Example: dropping rows with missing values

#Feature Scaling:
'''
Since clustering algorithms are sensitive to scale,
 we need to standardize the data to ensure that
 features with different ranges do not dominate
 the clustering.
'''
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 1:])  # Exclude non-numeric columns (like City)

# Convert scaled data back to a DataFrame for easy manipulation
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[1:])

#Determine the Optimal Number of Clusters (Elbow Method)
# List to store WCSS values for different cluster counts
wcss = []

# Try clustering with 1 to 10 clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()
'''Inference: Look for the "elbow" point in the plot.
This is the point where adding more clusters
 doesn’t significantly reduce WCSS, suggesting
 the optimal number of clusters.'''

#Apply K-means Clustering with Optimal Clusters
# Apply K-means with the optimal number of clusters (e.g., 4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Check the size of each cluster
print(df['Cluster'].value_counts())

#Visualize the Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Murder', y='Assault', hue='Cluster', data=df, palette='viridis', legend='full')
plt.title('Clusters of US Cities Based on Crime Rates')
plt.show()

'''
Analyze and Draw Inferences from the Clusters

Cluster 0: High murder and assault rates 
– cities with high violent crime rates.
Cluster 1: Moderate crime rates across all
 features – average-risk areas.
Cluster 2: Low crime rates – safe cities with
 minimal violent and property crimes.
Cluster 3: Cities with high rates of property 
crimes but relatively fewer violent crimes.
'''

