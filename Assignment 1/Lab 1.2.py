### Import packages in the virtual environment
import os  ## no need to install package0
import sys  ## no need to install package
import matplotlib
import pandas as pd  ## search the package "pandans"
import statsmodels.api as sm  ## search the package "statsmodels"

os.chdir(r"C:\Users\Asus\OneDrive\Desktop\Semester 2\Data Mining for business analytics\Assignment 1")
##Change the address to your working folder

pd.set_option('display.max_columns', 500)
## Display all columns when printing results

mydata = pd.read_csv("advertising.csv")
## Remember to include “r” first inside the bracket.

file = open('output.txt', 'wt')
sys.stdout = file
## Open a text file and save all results into the text file

print(mydata.describe())
## Display summary statistics of the data

df = pd.DataFrame(mydata)
##Transform the dataset into two-dimensional, size-mutable, potentially heterogeneous tabular data

######################## Linear regression with only one independent variable
x = df['Male']
##Define the independent variables in the model
y = df['ClickedonAd']
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model

model1 = sm.Logit(y, x).fit()
##Fit the model
results_model1 = model1.summary()
print(results_model1)
##Output the results


######################## Linear regression with only multiple independent variables

x = df[['DailyTimeSpentonSite', 'Age', 'AreaIncome', 'DailyInternetUsage', 'Male', 'Educationlevel', 'Workinghrsperweek']]
##Define the independent variables in the model
y = df[['ClickedonAd']]
##Define t he dependent variable in the model

# with statsmodels
x = sm.add_constant(x)  # adding a constant
##Add a constant in the regression model
model2 = sm.Logit(y, x).fit()
##Fit the model
results_model2 = model2.summary()
print(results_model2)
##Output the results


##file.close()
##Close the text file.
