# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv(r'E:\Campus\CodeAlpha\Sales Prediction\Advertising.csv')

# Exploratory Data Analysis (EDA)
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
data.info()
print("\nSummary statistics:")
print(data.describe())

# Checking for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Data Cleaning: Dropping rows with missing values
data = data.dropna()

# Visualizations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Feature Selection and Target Variable
X = data.drop(columns=['Sales'])
y = data['Sales']

# Encoding categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()

# Random Forest Regressor Model
rf_model = RandomForestRegressor(random_state=42)

# Combined Model using Voting Regressor
combined_model = VotingRegressor(estimators=[('Linear Regression', lr_model), ('Random Forest', rf_model)])
combined_model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred_combined = combined_model.predict(X_test)

print("Combined Model Performance:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_combined):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_combined):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred_combined)):.2f}")

# Feature Importance (for Random Forest)
rf_model.fit(X_train, y_train)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()
