# Exercise 3, task 2

import pandas as pd

df = pd.read_csv("../data/H1.csv")

# Filter out rows where 'IsCanceled' is 1 (canceled bookings)
df_non_canceled = df[df['IsCanceled'] == 0]

# Group by 'MarketSegment' and calculate total revenue from each segment ('ADR' (average daily rate) represents the
# revenue per room)
revenue_by_segment = df_non_canceled.groupby('MarketSegment')['ADR'].sum().reset_index()

# Sort the result by revenue in ascending order
revenue_by_segment = revenue_by_segment.sort_values(by='ADR')

print("Total income by market segment:" + "\n" + str(revenue_by_segment))