# Task 1

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("../data/life-expectancy-data.csv")

# Strip whitespace from column names
df.columns = [col.strip() for col in df.columns]

# List of columns to clean
wash_columns = ['Alcohol', 'percentage expenditure', 'BMI', 'Schooling', 'GDP', 'Life expectancy']

# Replace zero values with NaN and fill NaN with mean
for col in wash_columns:
    df[col] = df[col].replace(0, np.nan)
    mean = df[col].mean(skipna=True)
    df[col] = df[col].replace(np.nan, mean)

# Define features (X) and target (y)
y = df['Life expectancy']
X = df[['Alcohol', 'percentage expenditure', 'BMI', 'Schooling', 'GDP']]

print("R2 scores:" + "\n" + "-" * 10)

# Perform linear regression for each predictor column
for col in X.columns:
    x_i = X[col].values.reshape(-1, 1)  # Reshape data to 2D, as required by sklearn
    lr = LinearRegression()  # Create linear regression model
    lr.fit(x_i, y)  # Train the model on the current predictor (x_i) and the target (y)
    y_predictions = lr.predict(x_i)  # Predict life expectancy using the fitted model

    # Print R2 score for the current predictor
    # The value shows how well the model predicts the outcome of the depented variable
    # Value between 0 and 1, where 1 is a perfect prediction
    print(f"{col}: {r2_score(y, y_predictions):.3f}")
