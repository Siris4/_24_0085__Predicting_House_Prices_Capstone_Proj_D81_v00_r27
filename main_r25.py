import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import skew

# Load the dataset
data = pd.read_csv(r'C:\Users\Siris\Desktop\GitHub Projects 100 Days NewB\_24_0085__Day81_Predicting_House_Prices_Capstone_Proj__240902\NewProject\r00_env_START\boston.csv', index_col=0)

# Define the features (X) and the target (y)
X = data.drop(columns=['PRICE'])  # All columns except 'PRICE' are features
y = data['PRICE']  # 'PRICE' is the target variable

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate the predicted values and residuals
predicted_values = model.predict(X_train)
residuals = y_train - predicted_values

# Calculate the skewness of the residuals
skewness_residuals = skew(residuals)

# Determine how much the skewness is different from zero
difference_from_zero = abs(skewness_residuals)

# Output the results
print(f"Skewness of the residuals: {skewness_residuals:.4f}")
print(f"Difference from zero: {difference_from_zero:.4f}")
