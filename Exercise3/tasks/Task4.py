# Exercise 4, task 4

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/H1.csv")

# Filter out data for the year 2016
df_2016 = df[df['ArrivalDateYear'] == 2016]

# Group by 'ArrivalDateMonth'
df_2016['ArrivalDateMonth'] = pd.Categorical(df_2016['ArrivalDateMonth'],
                                             categories=['January', 'February', 'March', 'April', 'May',
                                                         'June', 'July', 'August', 'September', 'October',
                                                         'November', 'December'],
                                             ordered=True)


# Group by 'ArrivalDateMonth' and calculate the total number of cancellations and the average of the ADR
monthly_data = df_2016.groupby('ArrivalDateMonth').agg({'IsCanceled': 'sum', 'ADR': 'mean'}).reset_index()

# Plot the data
plt.figure(figsize=(10, 6))
# Plot the number of cancellations
plt.plot(monthly_data['ArrivalDateMonth'], monthly_data['IsCanceled'], marker='o', color='orange',
         label='Number of cancellations')
# Plot the average daily rate
plt.plot(monthly_data['ArrivalDateMonth'], monthly_data['ADR'], marker='o', color='pink', label='Average daily rate')

plt.title('Monthly ADR and cancellations in 2016', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of cancellations / ADR', fontsize=12)
# Add a legend
plt.legend()

# Rotate the x-axis labels
plt.xticks(rotation=45)
# Add a tight layout
plt.tight_layout()

plt.show()
