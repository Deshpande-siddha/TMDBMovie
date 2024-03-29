import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys    ## no need to install package
## Set working directory
os.chdir(r"C:\Users\Asus\OneDrive\Desktop\Semester 2\Data Mining for business analytics\Assignment 3")


## Import Data
mydata = pd.read_csv('clustering_results15.xlsx')


# Subset the dataset to cluster 1
cluster1_df = df[df['labels'] == 1]

# Calculate the mean sales for each type of jeans within cluster 1
mean_sales_cluster1 = cluster1_df.groupby('labels')[['Original', 'Fashion', 'Leisure', 'Stretch']].mean()

# Calculate the mean sales for each type of jeans across all stores
mean_sales_all = df[['Original', 'Fashion', 'Leisure', 'Stretch']].mean()

# Compare the mean sales for each type of jeans within cluster 1 to the mean sales across all stores
sales_comparison = mean_sales_cluster1 / mean_sales_all
