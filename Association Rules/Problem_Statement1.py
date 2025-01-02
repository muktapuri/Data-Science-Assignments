# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 02:44:51 2024

@author: Mukta
"""

'''
Problem Statement: -
Kitabi Duniya, a famous book store in India,
 which was established before Independence, the
 growth of the company was incremental year by
 year, but due to online selling of books and
 wide spread Internet access its annual growth
 started to collapse, seeing sharp downfalls,
 you as a Data Scientist help this heritage book
 store gain its popularity back and increase 
 footfall of customers and provide ways the 
 business can improve exponentially, apply 
 Association RuleAlgorithm, explain the rules,
 and visualize the graphs for clear understanding 
 of solution.
 
Business Objective:
Kitabi Duniya, a heritage bookstore, wants to 
increase customer footfall and revive its annual
 growth, which has been declining due to the 
 rise of online bookstores. The goal is to use
 data-driven strategies like association rule
 mining to identify customer preferences and 
 optimize book recommendations, promotions, and
 inventory management.
 
 
Constraints:
1.Limited access to customer demographics
 (assuming only transactional data is provided).
2.Maintaining the heritage value while adapting
 to modern retail strategies.
3.Budget constraints for implementing digital 
solutions.




'''

# Import required libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\Mukta\OneDrive\Desktop\Data_Science\Association Rules\book.csv"  # Replace with the correct path
df = pd.read_csv(file_path)

# --- Feature Engineering ---
# Step 1: Aggregate transactions for each customer (if 'Customer ID' is available)
if 'Customer ID' in df.columns:
    print("Aggregating transactions for each customer...")
    # Combine transactions by summing quantities for each customer
    df = df.groupby('Customer ID').sum()
    print("Transactions aggregated by customer.")

# Step 2: Create Transaction-Item Matrix
# Convert dataset into binary matrix (1 for purchased, 0 for not purchased)
print("\nCreating Transaction-Item Matrix...")
basket = df.applymap(lambda x: 1 if x > 0 else 0)
print("Transaction-Item Matrix created.")

# --- Apply Apriori Algorithm ---
# Find frequent itemsets with minimum support of 0.05
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Add a column indicating the length of each itemset
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

# Display the frequent itemsets
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# --- Generate Association Rules ---
# Create association rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Display a summary of the association rules
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# --- Visualize Results ---
# Scatter plot of support vs confidence
plt.figure(figsize=(8, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.6, edgecolors='r')
plt.title('Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True)
plt.show()

# Heatmap of antecedents vs consequents using lift
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Create a pivot table for heatmap plotting
pivot_table = rules.pivot(index='antecedents', columns='consequents', values='lift')

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(pivot_table, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Lift')
plt.title('Antecedents vs Consequents (Lift)')
plt.xlabel('Consequents')
plt.ylabel('Antecedents')
plt.xticks(range(len(pivot_table.columns)), pivot_table.columns, rotation=90)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)
plt.show()

'''
This code performs market basket analysis on a
 dataset using the Apriori algorithm to generate
 association rules. Here's a   steps:

Load Dataset:
Reads the dataset, which contains transaction-level
 data of book purchases.
 
Feature Engineering:

Transaction-Item Matrix: Converts raw transaction
 data into a binary matrix where rows represent
 transactions (or customers) and columns represent books. Entries are 1 if a book is purchased, otherwise 0.
Aggregate Transactions: Combines multiple
 transactions for the same customer into a single
 record if the dataset includes a Customer ID
 column.
Frequent Itemset Generation:

Uses the Apriori algorithm to identify frequently
 purchased book combinations, based on a minimum 
 support threshold.
Association Rule Mining:

Extracts association rules (e.g., "If book A is 
purchased, book B is likely to be purchased") with metrics such as support, confidence, and lift.

Visualization:

Scatter Plot: Displays the relationship between
 support and confidence for the rules.
Heatmap: Illustrates the lift metric between
 antecedents and consequents, helping visualize 
 the strength of relationships.
This code helps identify actionable insights,
 like recommending books to customers based on
 their purchasing patterns.
'''
