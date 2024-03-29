## Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier ##Install the package "scikit-learn"
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt
import os

os.chdir(r"C:\Users\Shuting Wang\Dropbox\Teaching\Baruch\Teaching2023 spring\CIS9660\Labs\Decision Tree")
## Set working directory


## Import Data
mydata = pd.read_csv("Titanic.csv")


## Determine features and target (class)
#split dataset in features and target variable
feature_cols = ['Pclass',	'Male',	'Age',	'SibSp',	'Parch',	'Fare']
target_col = ['Survived']
X = mydata[feature_cols] # Features
y = mydata[target_col] # Target variable

## Splitting the data into two parts: (1) a training set and (2) a test set.
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


## Build Decision Tree
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_split=3)
# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)


## Visualizing Decision Trees
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplot.png')
## You will see a .png file named decistion_tree.png being created in the main folder.

## Evaluating Model
# Predict the response for test dataset
y_pred1 = clf.predict(X_train)
y_pred2 = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of train dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred1))
print("Accuracy of test dataset(criterion=entropy, max_depth=4, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred2))



## Optimize Your Decision Tree
# Create Decision Tree classifer object – set the criterion to entropy and control the maximum depths
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_split=3)
##criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
##The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
##min_samples_split: The minimum number of samples required to split an internal node. Default=2


# Train Decision Tree Classifer
results = clf.fit(X_train,y_train)
#Plot the results
plt.figure(figsize=(40,20))
tree.plot_tree(results, feature_names = X.columns)
plt.savefig('treeplot2.png')
#Predict the response for test dataset
y_pred3 = clf.predict(X_train)
y_pred4 = clf.predict(X_test)

print("Accuracy of train dataset (criterion=entropy, max_depth=6, min_samples_split=3):",metrics.accuracy_score(y_train, y_pred3))
print("Accuracy of test dataset (criterion=entropy, max_depth=6, min_samples_split=3):",metrics.accuracy_score(y_test, y_pred4))

