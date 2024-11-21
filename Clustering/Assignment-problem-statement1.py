# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 16:52:51 2024

@author: Mukta
"""

'''
1.Problem Statement:
Perform K means clustering on the airlines
 dataset to obtain optimum number of clusters.
 Draw the inferences from the clusters
 obtained. Refer to EastWestAirlines.xlsx
 dataset.

1.1.What is the Business Objective?
The objective is to segment customers of the
 airline based on their behavior and other
 relevant attributes using K-Means clustering.
 The goal is to identify patterns among
 different customer segments to:

     1.Improve customer retention through
     personalized offers and loyalty programs.
     2.Optimize marketing strategies by
     targeting the right customers with
     relevant campaigns.
     3.Enhance customer experience by better
     understanding diverse customer needs.
     4.Identify high-value customers for
     priority services or exclusive offerings.
     
1.2.Are there any Constraints?
Cluster Interpretability: 
    The number of clusters should be optimal
    and meaningful for business decisions.

Data Quality: 
    Requires pre-processing (handling missing
    values, scaling, etc.) for K-means to work
    effectively.

Computational Constraints: 
    Clustering large datasets could be
    resource-intensive, impacting time or
    memory.

Fixed Dataset:
    Only the EastWestAirlines.xlsx dataset is
    available, so the analysis is constrained
    by the features provided.

Cluster Size Balance:
    If clusters are unevenly distributed,
    business decisions may require careful
    adjustment.

2. Data Dictionary

Name of Feature     	Description  	                                  Type	         Relevance
ID#                	Unique customer identifier	                      Quantitative,      Nominal	Irrelevant for clustering (ID does not contain useful information)
Balance	            Account balance of the customer	                  Quantitative,      Continuous	Relevant to analyze customer value
Qual_miles	        Number of qualifying miles earned	              Quantitative,      Discrete	Useful for understanding engagement
cc1_miles	        Credit card 1 miles category	                  Categorical,       Ordinal	Relevant for customer segmentation
cc2_miles	        Credit card 2 miles category	                  Categorical,       Ordinal	Relevant for customer segmentation
cc3_miles	        Credit card 3 miles category	                  Categorical,       Ordinal	Relevant for customer segmentation
Bonus_miles	        Number of bonus miles earned	                  Quantitative,      Discrete	Helps to evaluate customer loyalty
Bonus_trans	        Number of bonus transactions	                  Quantitative,      Discrete	Relevant for understanding transaction behavior
Flight_miles_12mo	Flight miles in the last 12 months	              Quantitative,      Continuous	Helps analyze recent flight activity
Flight_trans_12	    Number of flight transactions in last 12 months	  Quantitative,      Discrete	Relevant for recent engagement tracking
Days_since_enroll	Days since the customer enrolled	              Quantitative,      Continuous	Useful to track customer loyalty over time
Award?	            Award status (1 = Yes, 0 = No)	                  Categorical,       Binary	Relevant to understand loyalty program success

'''
#3.Exploratory Data Analysis (EDA):
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset 
df = pd.read_excel(r"C:\Users\Mukta\OneDrive\Desktop\DS\Dataset\EastWestAirlines.xlsx")

# Summary statistics of the dataset
summary_stats = df.describe()
print(summary_stats)
#Prints statistical information like mean, standard deviation, min, and max for numerical features.
'''ID#       Balance  ...  Days_since_enroll       Award?
count  3999.000000  3.999000e+03  ...         3999.00000  3999.000000
mean   2014.819455  7.360133e+04  ...         4118.55939     0.370343
std    1160.764358  1.007757e+05  ...         2065.13454     0.482957
min       1.000000  0.000000e+00  ...            2.00000     0.000000
25%    1010.500000  1.852750e+04  ...         2330.00000     0.000000
50%    2016.000000  4.309700e+04  ...         4096.00000     0.000000
75%    3020.500000  9.240400e+04  ...         5790.50000     1.000000
max    4021.000000  1.704838e+06  ...         8296.00000     1.000000

[8 rows x 12 columns]
'''


# 1. Univariate Analysis - Distribution of 'Balance'
plt.figure(figsize=(8, 5))
sns.histplot(df['Balance'], bins=30, kde=True, color='teal')
plt.title('Distribution of Account Balance')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

# 2. Bivariate Analysis - Scatter plot of 'Bonus_miles' vs 'Flight_miles_12mo'
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Bonus_miles', y='Flight_miles_12mo', data=df, color='purple')
plt.title('Bonus Miles vs Flight Miles (Last 12 Months)')
plt.xlabel('Bonus Miles')
plt.ylabel('Flight Miles in Last 12 Months')
plt.show()

# 3. Correlation Matrix Heatmap
corr_matrix = df.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

#4.Data Pre-processing 
#4.1 Data Cleaning, Feature Engineering, etc.
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Remove irrelevant columns
df_cleaned = df.drop(['ID#'], axis=1)

# 2. Check for missing values
print(df_cleaned.isnull().sum())  #  all 0 bcz no missing values

# 3. Feature Engineering: Create a new feature 'Total Credit Card Miles'
df_cleaned['Total_Credit_Card_Miles'] = df_cleaned[['cc1_miles', 'cc2_miles', 'cc3_miles']].sum(axis=1)

# 4. Scale numerical features for clustering
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_cleaned[['Balance', 'Bonus_miles', 'Days_since_enroll', 'Flight_miles_12mo', 'Total_Credit_Card_Miles']])

# Create a new DataFrame with scaled features
df_scaled = pd.DataFrame(scaled_features, columns=['Balance', 'Bonus_miles', 'Days_since_enroll','Flight_miles_12mo', 'Total_Credit_Card_Miles'])

# Include binary column 'Award?' back into the scaled dataset
df_scaled['Award?'] = df_cleaned['Award?'].values

# Display the cleaned and processed data
print(df_scaled.head())


#5.Model Building
#5.1 Perform K-means Clustering and Obtain Optimum Number of Clusters Using Scree Plot
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# List to store WCSS values for different numbers of clusters
wcss = []

# Try different cluster counts (from 1 to 10)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the scree plot (Elbow Method)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()

'''
5.2 Validate the Clusters and Derive Insights:
After determining the optimal number of clusters
 from the scree plot, we perform clustering and
 compare results with different numbers of clusters. We’ll assign cluster labels to the data and analyze each cluster to derive meaningful insights.
'''
# Build the K-means model with the optimal number of clusters (e.g., 3 clusters)
optimal_clusters = 3  # Replace this with the actual optimal number
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_scaled['Cluster'] = kmeans.fit_predict(df_scaled)

# Analyze the size of each cluster
print(df_scaled['Cluster'].value_counts())

# Add cluster labels to the original dataset
df_cleaned['Cluster'] = df_scaled['Cluster']

# Visualize the clusters with a scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Balance', y='Bonus_miles', hue='Cluster', data=df_cleaned, palette='viridis')
plt.title('Clusters based on Balance and Bonus Miles')
plt.show()

#6.Write about the benefits/impact of the solution 
#- in what way does the business (client) benefit
# from the solution provided? 
'''
The clustering solution empowers the airline to
 better understand its customers, leading to 
 personalized service, increased loyalty, and 
 cost optimization. By leveraging these insights,
 the business can build a customer-centric strategy
 that enhances both profitability and customer
 satisfaction, positioning the airline for sustained
 growth and competitive advantage.
'''

