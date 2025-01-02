# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:57:45 2025

@author: Mukta
"""

'''
Problem Statement: - 
A Mobile Phone manufacturing company wants to launch its three brand new phone into the market, but before going with its traditional marketing approach this time it want to analyze the data of its previous model sales in different regions and you have been hired as an Data Scientist to help them out, use the Association rules concept and provide your insights to the company’s marketing team to improve its sales.

 business objective:

The objective is to analyze previous model sales data across different regions to identify patterns and associations that can inform marketing strategies for the launch of three new phone models.

constraints:-

1.The dataset may contain missing values or inconsistencies.
2.The analysis must be completed within a limited timeframe.
3.The model should be interpretable for stakeholders.

Data Dictionary
Features:
Region: The geographical area where the sales occurred (e.g., North, South, East, West).
Phone Model: The specific model of the phone sold (e.g., Model A, Model B, Model C).
Sales: Binary values indicating whether a sale occurred (1) or not (0) for each phone model in each region.


'''
#Data Pre-processing
# Data Cleaning
import pandas as pd

# Load dataset
data = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\Data_Science\Association Rules\myphonedata.csv", header=None)

# Display the first few rows
print("Original Data:")
print(data.head())

# Check for missing values
print("Missing Values:")
print(data.isnull().sum())

#already encoded in format suitable for apriori
#apply apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets=apriori(data,min_support=0.2,use_colnames=True)
frequent_itemsets