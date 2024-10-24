# Exercise 3, task 1

import pandas as pd

df = pd.read_csv("../data/H1.csv")

# Create dataframe, group by country and include the 10 countries that occurs the most
top_countries = df['Country'].value_counts().head(10)

print("The top 10 countries with the most customers:" + "\n" + str(top_countries))