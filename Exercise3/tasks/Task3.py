# Exercise 3, task 3

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/H1.csv")

# Set the number of bins and the range for the histogram
bins = 40
adr_min, adr_max = df['ADR'].min(), df['ADR'].max()
print('Min ADR:', adr_min, 'Max ADR:', adr_max)

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(df['ADR'], bins=bins, range=(adr_min, adr_max), color='skyblue', edgecolor='black')
plt.title('Room Prices (ADR)', fontsize=14)
plt.xlabel('ADR (Average Daily Rate)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True)

plt.show()