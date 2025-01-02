# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:08:29 2025

@author: Mukta
"""
'''
Problem Statement: - 
A film distribution company wants to target audience based on their likes and dislikes, you as a Chief Data Scientist Analyze the data and come up with different rules of movie list so that the business objective is achieved.

 business objective:-

To analyze audience preferences for movies to create targeted marketing strategies and improve film distribution based on likes and dislikes.

 constraints:-

The dataset may have missing values or inconsistencies.
The analysis must be completed within a limited timeframe.
The model should be interpretable for stakeholders.

Data Dictionary
Features:
Movie Titles: Names of the movies (e.g., Sixth Sense, Gladiator, LOTR1, etc.)
Audience Preferences: Binary values indicating whether the audience liked (1) or disliked (0) each movie.


'''
#Data Pre-processing
#Data Cleaning
#Load the dataset and check for missing values.
import pandas as pd
# Load dataset
data = pd.read_csv(r"C:\Users\Mukta\OneDrive\Desktop\Data_Science\Association Rules\my_movies.csv", header=None)
# Define movie titles
movie_titles = ['Sixth Sense', 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot', 'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile']

# Assign movie titles to the columns
data.columns = movie_titles

# Display the first few rows to understand the structure
print("Original Data:")
print(data.head())

# Transpose the DataFrame so that each row represents a viewer's preferences
binary_preferences = data.T

# Set the first row as the header
binary_preferences.columns = binary_preferences.iloc[0]
binary_preferences = binary_preferences[1:]

# Check the structure of the DataFrame before conversion
print("Binary Preferences DataFrame before conversion:")
print(binary_preferences)

# Check for non-numeric values
print("Check for non-numeric values:")
print(binary_preferences.applymap(lambda x: isinstance(x, str)))

# Convert the DataFrame to integers
# Use pd.to_numeric with errors='coerce' to convert and handle non-numeric values
binary_preferences = binary_preferences.apply(pd.to_numeric, errors='coerce')

# Check for any NaN values that may have resulted from the conversion
print("Check for NaN values after conversion:")
print(binary_preferences.isnull().sum())

# Handle NaN values: Fill with 0 (assuming that a missing preference means 'disliked')
binary_preferences.fillna(0, inplace=True)

# Convert to integers
binary_preferences = binary_preferences.astype(int)

# Display the cleaned binary preference DataFrame
print("Cleaned Binary Preferences DataFrame:")
print(binary_preferences)

from mlxtend.frequent_patterns import apriori, association_rules

# Apply Apriori algorithm directly on the binary preference DataFrame
frequent_itemsets = apriori(binary_preferences, min_support=0.2, use_colnames=True)

# Display frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)


# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display the rules
print("Association Rules:")
print(rules)
