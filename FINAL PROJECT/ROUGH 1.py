import pandas as pd
import re
import statsmodels.api as sm
import sys
import os
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.api as sms
from pandas import Series, DataFrame
from statsmodels.compat import lzip
from datetime import datetime
from scipy.stats import shapiro
import json
import seaborn as sns
import matplotlib.pyplot as plt
#DATA CLEANING AND ORGANISING
credits_df = pd.read_csv("tmdb_5000_credits.csv")
credits_df.head()
movies_df = pd.read_csv("tmdb_5000_movies.csv")
movies_df.head()
# ### Cleaning Data
credits_df = credits_df.reindex(columns = ['movie_id', 'title', 'cast']) #removed crew
credits_df.head(1)
movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce').fillna(0) / 1000000
movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce').fillna(0) / 1000000
movies_df["release_date"] = pd.to_datetime(movies_df["release_date"])
movies_df["release_year"] = movies_df["release_date"].dt.year

# Create a new column called 'first_genre'
movies_df['first_genre'] = ""

# Extract the first genre from the 'genres' column for each row
for i in range(len(movies_df)):
    if isinstance(movies_df.loc[i, 'genres'], str):
        # Use regular expressions to extract the first genre from the 'genres' column
        match = re.findall('(?<=name\": \")[^\"]+', movies_df.loc[i, 'genres'])

        # Add the first genre to the new 'first_genre' column
        if len(match) > 0:
            movies_df.loc[i, 'first_genre'] = match[0]

# Convert 'first_genre' to a categorical column
movies_df['first_genre'] = pd.Categorical(movies_df['first_genre'])
movies_df.head(1)
movies_df = movies_df.reindex(columns = ['first_genre', 'title', 'vote_average', 'revenue', 'budget', 'popularity', 'runtime', 'release_year']) #removed unnecessary columns
movies_df.head(1)
movies_df.to_csv('movies_cleaned.csv', index=False)
#----------------------------------------------------------------------------------------------------------------------

#REGRESSION ANALYSIS
# Load dataset
movies_df = pd.read_csv('movies_cleaned.csv')
## Remember to include “r” first inside the bracket.
# Clean the dataset
movies_df = movies_df.dropna()  # Drop rows with missing values
movies_df = movies_df.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
movies_df = movies_df.dropna()  # Drop rows with NaN values

file = open('output.txt','wt')
sys.stdout = file
## Open a text file and save all results into the text file

print(movies_df.describe())
## Display summary statistics of the data

df = pd.DataFrame(movies_df)
##Transform the dataset into two-dimensional, size-mutable, potentially heterogeneous tabular data

######################## Linear regression with only one independent variable
x = df['revenue']
##Define the independent variables in the model
y = df['vote_average']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model1 = sm.OLS(y, x).fit()
##Fit the model
results_model1 = model1.summary()
print(results_model1)
##Output the results

df["predicted1"] = model1.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=df, x="revenue", y="predicted1")
plt.show()
##Plot the predicted value against the actual value of charges.



######################## Linear regression with only multiple independent variables

x = df[['revenue', 'budget', 'popularity', 'runtime']]
##Define the independent variables in the model
y = df['vote_average']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model2 = sm.OLS(y, x).fit()
##Fit the model
results_model2 = model2.summary()
print(results_model2)
##Output the results

df["predicted2"] = model2.predict(x)
##Caculate the predicted value of dependent variable based on the model

sns.scatterplot(data=df, x="vote_average", y="predicted2")
plt.show()
##Plot the predicted value against the actual value of charges.

##Regression diagnostic I: multicollinearity
corr_matrix = df.corr()
print(corr_matrix)

##Regression diagnostic 2: heteroscedasticity in the error term
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model2.resid, model2.model.exog)
print(lzip(names, test))

##Regression diagnostic 3: autocorrelation
##This is not a time-series dataset. Thus, the concern about autocorrelation is minimul.

##Regression diagnostic 4: model specification errors
##1)The model does not exclude any “core” variables.
##2)The model does not include superfluous variables.
##3)The functional form of the model is suitably chosen.
##4)There are no errors of measurement in the regressand and regressors.
##5)Outliers in the data, if any, are taken into account.

print(movies_df.describe())

##6)The probability distribution of the error term is well specified.
residuals = model2.resid
print(shapiro(residuals))

file.close()
##Close the text file.
#---------------------------------------------------------------------------------------
#DATA VISUALIZATION
#Heatmap of correlations
# assuming your dataset is stored in a pandas dataframe called 'df'
corr_matrix = df.corr()

# create heatmap using seaborn
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# set title
plt.title('Correlation Heatmap')

# display the plot
plt.show()

#Bubble plot
# Load the data into a pandas DataFrame
df = pd.read_csv("movies_cleaned.csv")

# Create a scatter plot with revenue on the x-axis, vote_average on the y-axis, and bubble size indicating popularity
plt.scatter(df['popularity'], df['vote_average'], s=df['popularity']*10, alpha=0.5)

# Set the x-axis label
plt.xlabel('popularity')

# Set the y-axis label
plt.ylabel('Vote Average')

# Set the plot title
plt.title('Popularity vs. Vote Average')

# Display the plot
plt.show()
