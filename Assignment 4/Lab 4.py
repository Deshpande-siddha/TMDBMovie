## Load libraries
import pandas as pd  ##Install package Pandas
import csv  ##No package to install
from mlxtend.preprocessing import TransactionEncoder   ##Install package mlxtend
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import os

## Set working directory
os.chdir(r"C:\Users\Asus\OneDrive\Desktop\Semester 2\Data Mining for business analytics\Assignment 4")


## Import Data
# Read CSV file and convert to list of lists
with open('Groceries2.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = []
    for row in reader:
        data.append(row[0].split(','))


## Display data
print(data)


##TransactionEncoder is a class in the Python mlxtend library that is used to transform transaction data in a tabular format into a one-hot encoded NumPy array suitable for use with the apriori algorithm.
##The TransactionEncoder class takes as input a list of transactions, where each transaction is represented as a list of items. It then fits the encoder to the data, identifying all unique items across all transactions and creating a mapping from each item to a unique integer identifier.
## Finally, it transforms the data into a one-hot encoded NumPy array, where each row represents a transaction and each column represents a unique item, with a value of 1 indicating that the item was present in the transaction and a value of 0 indicating that it was not.

te = TransactionEncoder()
te_array = te.fit_transform(data)
df = pd.DataFrame(te_array, columns=te.columns_)
print(df)

# find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

# print top 5 rules
print(rules.head())


# Save the association rules to an Excel file
rules.to_excel("association_rules.xlsx", index=False)