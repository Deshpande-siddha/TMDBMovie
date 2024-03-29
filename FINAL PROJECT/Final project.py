import pandas as pd
import numpy as np
import ast
import json
import string
import re
import nltk
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import statsmodels.api as sm
import sys
import os
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.st
import pandas as pd
import numpy as np
import ast
import json
import string
import re
import nltk
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import statsmodels.api as sm
import sys
import os
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
from pandas import Series, DataFrame
from statsmodels.compat import lzip
from datetime import datetime
from scipy.stats import shapiro
import json
import seaborn as sns
import matplotlib.pyplot as plt
import ast

import sys
import os
os.chdir(r'C:\Users\Asus\OneDrive\Desktop\Semester 2\Data Mining for business analytics\FINAL PROJECT')
# Read CSV files into primary dataframes

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
# Read CSV files into primary dataframes

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
df1
#DATA CLEANING
# Observe our dataset dimensions

print(df1.shape)
print(df2.shape)
#Cleaning Df1
#Unpacking “packed” fields: 'cast' and 'crew'
# Convert strings to lists of dictionaries

df1["cast"] = df1["cast"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Create new column in df1 - 'cast_names'

df1["cast_names"] = df1["cast"].apply(lambda x: [d["name"] for d in x])
#Note: We have used the ast.literal_eval() which is a Python built-in function that evaluates a string containing a literal (such as a dictionary, list, tuple, number, or boolean) and returns the corresponding Python object. Also used a lambda function to iterate over each element of the "cast" column (which is a list of dictionaries) and extracted the value associated with the key "name" from each dictionary, creating a new list of names.

df1
# Examine members of the crew list

crew_list = json.loads(df1['crew'][1])
for crew_member in crew_list:
    print(crew_member['job'])
#A JSON (JavaScript Object Notation) object is a data structure that stores data in a text format using a collection of key-value pairs.

#Here, the json.loads() function is used to convert the string representation of a JSON object stored in the 'crew' column of dataframe df1 into a Python list of dictionaries, which is assigned to the variable crew_list.Then, a loop is iterated over each dictionary in crew_list, and the value associated with the key 'job' is printed for each dictionary. This code has been used to extract and display the 'job' information for each crew member in the 'crew' column.

# Apply literal_eval to convert stringified dictionaries to dictionaries (ensuring that our data is in dictionary format)

df1["crew"] = df1["crew"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Extract names of director, producer, and screenplay writer from crew list

df1["director"] = df1["crew"].apply(lambda x: [d["name"] for d in x if d["job"] == "Director"][0] if [d["job"] for d in x if d["job"] == "Director"] else None)
df1["producer"] = df1["crew"].apply(lambda x: [d["name"] for d in x if d["job"] == "Producer"])
df1["screenplay_writer"] = df1["crew"].apply(lambda x: [d["name"] for d in x if d["job"] == "Screenplay"])

#Reasons for including the following crew members:

#Director: The director has a significant impact on the overall vision and style of a movie, including the tone, pacing, camera work, and performance direction. Movies directed by the same director may have similar themes, visual styles, or narrative techniques, which can be used for recommendation purposes.

#Screenplay writer: The screenplay is the foundation of a movie, providing the story, characters, dialogue, and structure. Similarities between movies based on the same source material or with similar themes, genres, or narrative structures can be identified and used for recommendation.

#Producer: The producer oversees the financial and logistical aspects of a movie, including casting, hiring, scheduling, and marketing. The production company or studio associated with a movie may have a specific brand or target audience, which can be used for recommendation purposes. Additionally, producers may have a track record of successful movies or collaborations with specific directors or actors, which can also be used as a recommendation feature.

df1
# Examine empty values in the directors column

df1[df1['director'].isnull()]['title']
df1.info()
# Drop null values from the "director" column
df1.dropna(subset=['director'], inplace=True)

# Verify the changes
print(df1.info())
# describe NaN and empty cells in each column

for col in df1.columns:
    nan_count = df1[col].isna().sum()
    empty_count = df1[col].eq('').sum()
    print(f"Column {col}: NaN count = {nan_count}, Empty count = {empty_count}")
df1
df1 = df1.dropna()  # Drop rows with missing values
df1 = df1.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
df1 = df1.dropna()  # Drop rows with NaN values
df1= df1.reindex(columns = ['movie_id', 'title', 'cast_names', 'director'])
df1
#Seperating Cast into Top 3

df1[['cast1', 'cast2', 'cast3']] = df1['cast_names'].apply(lambda x: (x[0], x[1], x[2]) if len(x)>=3 else [None, None, None]).apply(pd.Series)
df1
df1 = df1.drop('cast_names', axis=1)
df1
#Cleaning Df2
df2["release_date"] = pd.to_datetime(df2["release_date"])
df2["release_year"] = df2["release_date"].dt.year
df2['first_genre'] = ""
# Extract the first genre from the 'genres' column for each row
for i in range(len(df2)):
    if isinstance(df2.loc[i, 'genres'], str):
        # Use regular expressions to extract the first genre from the 'genres' column
        match = re.findall('(?<=name\": \")[^\"]+', df2.loc[i, 'genres'])

        # Add the first genre to the new 'first_genre' column
        if len(match) > 0:
            df2.loc[i, 'first_genre'] = match[0]
df2['first_genre'] = pd.Categorical(df2['first_genre'])
df2 = df2.dropna()  # Drop rows with missing values
df2 = df2.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
df2 = df2.dropna()  # Drop rows with NaN values
df2.drop_duplicates(inplace=True)
df2.head(1)
df2["profit"] = df2["revenue"] - df2["budget"] #created profit column
df2 = df2.reindex(columns = ['id', 'first_genre', 'title', 'vote_average', 'revenue', 'budget', 'profit', 'popularity', 'runtime', 'release_year']) #removed unnecessary columns
df2
movies = df2.merge(df1,on="title")
movies = movies.drop("movie_id", axis=1)
movies

######################## Linear regression with only one independent variable
x = movies['revenue']
##Define the independent variables in the model
y = movies['vote_average']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model1 = sm.OLS(y, x).fit()
##Fit the model
results_model1 = model1.summary()
print(results_model1)
movies["vote_average"] = model1.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=movies, x="vote_average", y="revenue")
plt.show()
##Plot the predicted value against the actual value of charges.
movies["vote_average"] = model1.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=movies, x="vote_average", y="revenue")
plt.show()
##Plot the predicted value against the actual value of charges.
######################## Linear regression with only multiple independent variables

x = movies[['vote_average', 'budget', 'popularity', 'runtime']]
##Define the independent variables in the model
y = movies['revenue']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model2 = sm.OLS(y, x).fit()
##Fit the model
results_model2 = model2.summary()
print(results_model2)
##Output the results

movies["predicted2"] = model2.predict(x)
##Caculate the predicted value of dependent variable based on the model

#sns.scatterplot(data=movies, x="vote_average", y="revenue")
#plt.show()
##Plot the predicted value against the actual value of charges.
##Regression diagnostic I: multicollinearity
corr_matrix = movies.corr()
print(corr_matrix)
##Regression diagnostic I: multicollinearity
corr_matrix = movies.corr()
print(corr_matrix)

##Regression diagnostic 2: heteroscedasticity in the error term
#names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
#test = sms.het_breuschpagan(model2.resid, model2.model.exog)
#print(lzip(names, test))

##Regression diagnostic 3: autocorrelation
##This is not a time-series dataset. Thus, the concern about autocorrelation is minimul.

##Regression diagnostic 4: model specification errors
##1)The model does not exclude any “core” variables.
##2)The model does not include superfluous variables.
##3)The functional form of the model is suitably chosen.
##4)There are no errors of measurement in the regressand and regressors.
##5)Outliers in the data, if any, are taken into account.

#print(movies.describe())

##6)The probability distribution of the error term is well specified.
#residuals = model2.resid
#print(shapiro(residuals))
corr_matrix = movies.corr()

# create heatmap using seaborn
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# set title
plt.title('Correlation Heatmap')

# display the plot
plt.show()
corr_matrix = movies.corr()

# create heatmap using seaborn
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# set title
plt.title('Correlation Heatmap')

# display the plot
plt.show()

#
#Bubble plot

# Create a scatter plot with revenue on the x-axis, vote_average on the y-axis, and bubble size indicating popularity
#plt.scatter(movies['popularity'], movies['vote_average'], s=movies['popularity']*10, alpha=0.5)

# Set the x-axis label
#plt.xlabel('popularity')

# Set the y-axis label
#plt.ylabel('Vote Average')

# Set the plot title
#plt.title('Popularity vs. Vote Average')

# Display the plot
#plt.show()
#Analysis 2 - Logit

movies['profit'] = movies['profit'].astype(int)
movies
######################## Logit regression with only one independent variable
x = movies['popularity']
##Define the independent variables in the model
y = movies['profit_true']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model
model1 = sm.Logit(y, x).fit()
##Fit the model
results_model1 = model1.summary()
print(results_model1)
##Output the results

# Separate the data for profitable and unprofitable movies
popularity_profitable = movies[movies['profit_true'] == 1]['popularity']
popularity_unprofitable = movies[movies['profit_true'] == 0]['popularity']

# Create the KDE plot
sns.kdeplot(popularity_profitable, label='Profitable Movies')
sns.kdeplot(popularity_unprofitable, label='Unprofitable Movies')

# Set plot title and labels
plt.title('Probability Density Plot of Popularity')
plt.xlabel('Popularity')
plt.ylabel('Density')

# Display the plot
plt.show()

######################## Linear regression with only multiple independent variables

x = movies[['popularity', 'budget', 'vote_average']]
##Define the independent variables in the model
y = movies['profit_true']
##Define the dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model2 = sm.Logit(y, x).fit()
##Fit the model
results_model2 = model2.summary()
print(results_model2)
#Analysis 3 - Decision Tree

movies['profit']
movies['profit_true'] = movies['profit'].apply(lambda x: 1 if x > 0 else 0)
movies['profit_true']
movies
## Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier ##Install the package "scikit-learn"
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt
import os

## Determine features and target (class)
#split dataset in features and target variable
feature_cols = ['budget','vote_average','popularity','runtime']
target_col = ['profit_true']
X = movies[feature_cols] # Features
y = movies[target_col] # Target variable

## Splitting the data into two parts: (1) a training set and (2) a test set.
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1) # 70% training and 30% test

## Build Decision Tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_split=3)
# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)
​
​
## Visualizing Decision Trees
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
#plt.savefig('treeplot.png')
## You will see a .png file named decistion_tree.png being created in the main folder.
​
## Evaluating Model
# Predict the response for test dataset
y_pred1 = clf.predict(X_train)
y_pred2 = clf.predict(X_test)
​
# Model Accuracy, how often is the classifier correct?
print("Accuracy of train dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred1))
print("Accuracy of test dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred2))
​
​
​
## Optimize Your Decision Tree
# Create Decision Tree classifer object – set the criterion to entropy and control the maximum depths
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=3)
##criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
##The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
##min_samples_split: The minimum number of samples required to split an internal node. Default=2
​
​
# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)
#Plot the results
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
#plt.savefig('treeplot2.png')
#Predict the response for test dataset
y_pred3 = clf.predict(X_train)
y_pred4 = clf.predict(X_test)
​
print("Accuracy of train dataset (criterion=entropy, max_depth=6, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred3))
print("Accuracy of test dataset (criterion=entropy, max_depth=6, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred4))
​
movies.info()
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for association rule mining
selected_cols = ['first_genre', 'director', 'cast1', 'cast2', 'cast3']
assoc_df = movies[selected_cols]
​
# Remove rows with missing values
assoc_df = assoc_df.dropna().copy()
​
# Convert the dataframe to a transactional format
transactions = []
for _, row in assoc_df.iterrows():
    transaction = []
    for col in selected_cols:
        transaction.append(row[col])
    transactions.append(transaction)
​
# Apply one-hot encoding to convert the transactional format
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)
​
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot_df, min_support=0.001, use_colnames=True)
​
# Generate association rules with lower confidence threshold
association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.03)
​
# Display the association rules
print(association_rules)
​
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for association rule mining
selected_cols = ['director', 'cast1', 'cast2', 'cast3', 'first_genre']
assoc_df = movies[selected_cols]
​
# Remove rows with missing values
assoc_df = assoc_df.dropna().copy()
​
# Convert the dataframe to a transactional format
transactions = []
for _, row in assoc_df.iterrows():
    transaction = []
    for col in selected_cols:
        transaction.append(row[col])
    transactions.append(transaction)
​
# Apply one-hot encoding to convert the transactional format
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)
​
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot_df, min_support=0.001, use_colnames=True)
​
# Generate association rules with the specific genre as antecedent
association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
genre_of_interest = 'Action'  # Replace with the genre you are interested in
filtered_rules = association_rules[association_rules['antecedents'].apply(lambda x: genre_of_interest in x)]
​
# Sort the rules by confidence
filtered_rules = filtered_rules.sort_values(by='confidence', ascending=False)
​
# Display the association rules
print(filtered_rules)
​
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for analysis
selected_cols = ['director', 'cast1', 'cast2', 'cast3', 'profit']
​
# Filter out rows with missing values
filtered_df = movies[selected_cols].dropna().copy()
​
# Split the data into features (directors and casts) and target (profit)
features = filtered_df[['director', 'cast1', 'cast2', 'cast3']]
target = filtered_df['profit']
​
# Perform one-hot encoding on the categorical features
features_encoded = pd.get_dummies(features)
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)
​
# Create a decision tree regressor
regressor = DecisionTreeRegressor()
​
# Fit the model on the training data
regressor.fit(X_train, y_train)
​
# Predict the profits for the test data
predictions = regressor.predict(X_test)
​
# Combine the predicted profits with the corresponding features
results_df = X_test.copy()
results_df['predicted_profit'] = predictions
​
# Sort the results by predicted profit in descending order
results_df = results_df.sort_values(by='predicted_profit', ascending=False)
​
# Display the top combinations with highest predicted profit
top_combinations = results_df.head(10)
top_combinations
​
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for analysis
selected_cols = ['director', 'cast1', 'cast2', 'cast3', 'profit']
​
# Filter out rows with missing values
filtered_df = movies_df[selected_cols].dropna().copy()
​
# Encode categorical variables using one-hot encoding
encoded_df = pd.get_dummies(filtered_df, columns=['director', 'cast1', 'cast2', 'cast3'])
​
# Split the data into features (directors and casts) and target (profit)
features = encoded_df.drop('profit', axis=1)
target = encoded_df['profit']
​
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
​
# Create a decision tree regressor
regressor = DecisionTreeRegressor()
​
# Fit the model on the training data
regressor.fit(X_train, y_train)
​
# Predict the profit for the testing data
predictions = regressor.predict(X_test)
​
# Combine the predicted profit with the corresponding features
results_df = X_test.copy()
results_df['predicted_profit'] = predictions
​
# Sort the results by predicted profit in descending order
results_df = results_df.sort_values(by='predicted_profit', ascending=False)
​
# Display the top combinations with highest predicted profit
top_combinations = results_df.head(10)
print(top_combinations)
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for analysis
selected_cols = ['cast1', 'cast2', 'cast3', 'revenue']
​
# Filter out rows with missing values
filtered_df = movies[selected_cols].dropna().copy()
​
# Group the data by actor and calculate the sum of revenue for each actor
actor_revenue = filtered_df.groupby(['cast1', 'cast2', 'cast3'])['revenue'].sum().reset_index()
​
# Sort the actors by revenue in descending order
top_actor = actor_revenue.sort_values(by='revenue', ascending=False).head(1)
​
# Display the actor with the highest revenue
print(top_actor)
​
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for association rule mining
selected_cols = ['director', 'cast1', 'cast2', 'cast3']
​
# Remove rows with missing values
assoc_df = movies[selected_cols].dropna().copy()
​
# Convert the dataframe to a transactional format
transactions = []
for _, row in assoc_df.iterrows():
    transaction = []
    for col in selected_cols:
        transaction.append(row[col])
    transactions.append(transaction)
​
# Apply one-hot encoding to convert the transactional format
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)
​
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot_df, min_support=0.001, use_colnames=True)
​
# Generate association rules with specified antecedents and consequents
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3,
                          antecedents={'director_Name'}, consequents={'cast1_Name', 'cast2_Name'})
​
# Display the association rules
print(rules)
​
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
​
# Assuming you have loaded the dataframe into a variable called 'movies_df'
​
# Select relevant columns for association rule mining
selected_cols = ['director', 'cast1', 'cast2', 'cast3']
​
# Remove rows with missing values
assoc_df = movies[selected_cols].dropna().copy()
​
# Convert the dataframe to a transactional format
transactions = []
for _, row in assoc_df.iterrows():
    transaction = []
    for col in selected_cols:
        transaction.append(row[col])
    transactions.append(transaction)
​
# Apply one-hot encoding to convert the transactional format
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)
​
# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(one_hot_df, min_support=0.001, use_colnames=True)
​
# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
​
# Filter the rules based on desired director and actors
director_name = "John Doe"
actor_names = ["Actor A", "Actor B", "Actor C"]
filtered_rules = rules[
    (rules['antecedents'].astype(str).str.contains(director_name)) &
    (rules['consequents'].astype(str).str.contains('|'.join(actor_names)))
]
​
# Display the filtered association rules
print(filtered_rules)
​