# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:28:49 2025

@author: Mukta
"""

'''
Problem Statement: - 
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.

Business Objective:-
The objective is to identify relationships between products sold at the departmental store using association rule mining. These insights will help:
    
1. Optimize product placement.
2. Create bundled offers to increase sales.
3. Enhance customer satisfaction by offering relevant product recommendations.
 
Constraints:-
1. Data Quality: Incomplete or inconsistent transactional data may require extensive preprocessing.
2. Scalability: Algorithms must handle large datasets efficiently.
3. Thresholds: Defining appropriate thresholds for support and confidence to balance meaningfulness and granularity of the rules.
4. Interpretability: Insights must be actionable and comprehensible to stakeholders. 
'''

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

#  Load Dataset
df = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\Data_Science\Association Rules\groceries.csv", on_bad_lines='skip')

# Display the first few rows of the dataset
print(df.head())

# Check dataset structure and data types
print("\nDataset Info:")
print(df.info())

# Data Preprocessing
# Remove any missing or null values if present
print("\nChecking for missing values:")
print(df.isnull().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Confirm there are no missing values left
print("\nAfter cleaning, missing values:")
print(df.isnull().sum())

# Standardize column names for consistency
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\nUpdated column names:")
print(df.columns)

#step1:convert the dataset into a format suitable for Apriori
te=TransactionEncoder()
te_ary=te.fit(df).transform(df)
df1=pd.DataFrame(te_ary,columns=te.columns_)
frequent_itemsets=apriori(df1,min_support=0.03,use_colnames=True)
print(frequent_itemsets)
df1

rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)
#try for threshold=2
#step4:Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])