# Loading the csv file
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys    ## no need to install package


## Set working directory
os.chdir(r"C:\Users\Asus\OneDrive\Desktop\Semester 2\Data Mining for business analytics\FINAL PROJECT")


## Import Data
mydata = pd.read_csv('tmdb_5000_movies.csv')
data=mydata.copy()
# Cleaning Data

# Identifying and cleaning outliers in the 'popularity' column
popularity = data['popularity']
qnt = popularity.quantile([0.25, 0.75])
H = 1.5 * (qnt[0.75] - qnt[0.25])
outliers_popularity = (popularity < (qnt[0.25] - H)) | (popularity > (qnt[0.75] + H))
popularity[outliers_popularity] = pd.NA
print("Outliers in popularity column:", outliers_popularity.to_numpy().nonzero()[0])

# Converting popularity row values into whole numbers
data['popularity'] = data['popularity'].round()

# Identifying and cleaning outliers in the 'runtime' column
runtime = data['runtime']
qnt = runtime.quantile([0.25, 0.75])
H = 1.5 * (qnt[0.75] - qnt[0.25])
outliers_runtime = (runtime < (qnt[0.25] - H)) | (runtime > (qnt[0.75] + H))
runtime[outliers_runtime] = pd.NA
print("Outliers in runtime column:", outliers_runtime.to_numpy().nonzero()[0])

# Identifying and cleaning outliers in the 'revenue' column
revenue = data['revenue']
qnt = revenue.quantile([0.25, 0.75])
H = 1.5 * (qnt[0.75] - qnt[0.25])
outliers_revenue = (revenue < (qnt[0.25] - H)) | (revenue > (qnt[0.75] + H))
revenue[outliers_revenue] = pd.NA
print("Outliers in revenue column:", outliers_revenue.to_numpy().nonzero()[0])

# Identifying and cleaning outliers in the 'vote_average' column
vote_average = data['vote_average']
qnt = vote_average.quantile([0.25, 0.75])
H = 1.5 * (qnt[0.75] - qnt[0.25])
outliers_vote_average = (vote_average < (qnt[0.25] - H)) | (vote_average > (qnt[0.75] + H))
vote_average[outliers_vote_average] = pd.NA
print("Outliers in vote_average column:", outliers_vote_average.to_numpy().nonzero()[0])

# Identifying and cleaning outliers in the 'vote_count' column
vote_count = data['vote_count']
qnt = vote_count.quantile([0.25, 0.75])
H = 1.5 * (qnt[0.75] - qnt[0.25])
outliers_vote_count = (vote_count < (qnt[0.25] - H)) | (vote_count > (qnt[0.75] + H))
vote_count[outliers_vote_count] = pd.NA
print("Outliers in vote_count column:", outliers_vote_count.to_numpy().nonzero()[0])

# Replacing the original columns in the dataframe with cleaned columns
data['runtime'] = runtime
data['revenue'] = revenue
data['vote_average'] = vote_average
data['vote_count'] = vote_count

# Create a new column called 'first_genre'
data['first_genre'] = ""

# Extract the first genre
# Extract the first genre from the 'genres' column for each row
for i in range(len(data)):
    # Use regular expressions to extract the first genre from the 'genres' column
    match = re.findall('(?<=name\": \")[^\"]+', data['genres'][i])

    # Add the first genre to the new 'first_genre' column
    if match:
        data.at[i, 'first_genre'] = match[0]

# Convert 'first_genre' to a categorical variable
data['first_genre'] = pd.Categorical(data['first_genre'])

# Drop all rows with null values in the 'popularity' column
data.dropna(subset=['popularity'], inplace=True)

# Cleaning budget column by reformatting values in $ millions
data['budget_millions'] = '$' + round(data['budget'] / 1000000).astype(str)
data['budget_millions'].isna().sum()

# Remove the dollar sign and convert the column to numeric
data['budget_millions'] = pd.to_numeric(data['budget_millions'].str.replace('$', ''))

# Cleaning revenue column by reformatting values in $ millions
data['revenue'] = pd.to_numeric(mydata_copy['revenue'], errors='coerce')
data['revenue_millions'] = '$' + round(data['revenue'] / 1000000)
data['revenue_millions'].isna().sum()

# Removing the dollar sign and convert the column to numeric
data['revenue_millions'] = pd.to_numeric(data['revenue_millions'].str.replace('$', ''))

# Removing rows with missing values in the 'budget_millions' and 'revenue_millions' columns
data.dropna(subset=['budget_millions', 'revenue_millions'], inplace=True)

# Removing rows with zero values in the 'budget_millions' and 'revenue_millions' columns
data = data[(data['budget_millions'] != 0) & (data['revenue_millions'] != 0)]

# Verifying the column data type and values
data['revenue_millions'].dtype
data['revenue_millions'].head()

# Removing all rows with $0 and NA values in the revenue_millions column
data = data[data['revenue_millions'] != '$0']
data = data[data['revenue_millions'] != '$NA']

# Removing all rows with 0 values in the runtime column
data = data[data['runtime'] != '0']

# Only keeping 'Released' movies from our dataset and dropping the unreleased ones
data = data[data['status'] == 'Released']

# Cleaning release_date column to keep just the year
data['release_year'] = data['release_date'].str[:4]

# Dropping irrelevant columns from the data
data = data.drop(columns=['budget', 'genres', 'homepage', 'id', 'keywords',
                          'original_language', 'overview', 'original_title',
                          'production_companies', 'production_countries', 'release_date',
                          'revenue', 'spoken_languages', 'status', 'tagline'])

data = data.drop(columns=['vote_count'])

data.head()
