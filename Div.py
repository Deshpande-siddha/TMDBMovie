import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data (replace 'data.csv' with your actual data file)
data = pd.DataFrame({
    "Summated_Rating": [51, 70, 69, 65, 65, 58, 56, 64, 52, 47, 53, 51, 76, 65, 55, 51, 54, 46, 55, 54, 63, 62, 55, 75, 69],
    "Cost_($_per_person)": [38, 52, 58, 60, 45, 39, 43, 44, 34, 31, 43, 39, 85, 54, 37, 34, 34, 22, 42, 45, 62, 59, 25, 46, 48]
})

# Perform linear regression
x = data["Summated_Rating"]
y = data["Cost_($_per_person)"]

x = sm.add_constant(x)  # Add a constant in the regression model

model = sm.OLS(y, x).fit()  # Fit the model

# Print the regression results
results = model.summary()
print(results)

# Calculate the predicted values
data["predicted"] = model.predict(x)

# Plot the predicted values against the actual values
sns.scatterplot(data=data, x="Summated_Rating", y="Cost_($_per_person)")
plt.show()
