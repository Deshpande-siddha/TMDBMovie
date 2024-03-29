import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('data.csv')

# Split the data into features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)