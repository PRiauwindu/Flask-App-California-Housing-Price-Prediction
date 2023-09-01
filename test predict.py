# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:16:48 2023

@author: putra
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model (you can use more advanced evaluation techniques)
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"Train R2 score: {train_score:.2f}")
print(f"Test R2 score: {test_score:.2f}")

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# Test Set

feature_data = X_test[3,]

# Create a DataFrame
# Convert the feature_data dictionary to a DataFrame
feature_df = pd.DataFrame(feature_data)
feature_df = feature_df.T
feature_df = scaler.transform(feature_df)

# Predict using the model
prediction = model.predict(feature_df)

# Print the prediction
print(prediction)
