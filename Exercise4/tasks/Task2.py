# Task 2

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("../data/life-expectancy-data.csv")

# Strip whitespace from column names
df.columns = [col.strip() for col in df.columns]

# List of columns to clean
wash_columns = ['Life expectancy', 'Year']

# Replace zero values with NaN and fill NaN with mean
for col in wash_columns:
    df[col] = df[col].replace(0, np.nan)
    mean = df[col].mean(skipna=True)
    df[col] = df[col].replace(np.nan, mean)

# Get a list of unique countries
unique_countries = df['Country'].unique()

# Create a dictionary to store the life expectancy of each country in 2020
life_expectancy_of_each_country_2020 = {}

# Loop through each country and predict the life expectancy in 2020
for country in unique_countries:
    country_data = df[df['Country'] == country]  # Filter data for the current country

    # Define the target variable (y) and the predictor (X)
    y = country_data['Life expectancy']
    X = country_data[['Year']].values.reshape(-1, 1)

    # Train a linear regression model
    lr = LinearRegression()
    lr.fit(X, y)

    # Predict the life expectancy in 2020 and store the result in the dictionary
    life_expectancy_of_each_country_2020[country] = lr.predict([[2020]])[0]

# Find the country with the highest life expectancy in 2020
country_with_best_life_expectancy = max(life_expectancy_of_each_country_2020,
                                        key=life_expectancy_of_each_country_2020.get)
life_expectancy_in_best_country = life_expectancy_of_each_country_2020[country_with_best_life_expectancy]

print(f"The country with the best life expectancy in 2020 is {country_with_best_life_expectancy} with a life expectancy "
      f"of {life_expectancy_in_best_country:.1f} years.")