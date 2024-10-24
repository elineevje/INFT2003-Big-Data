# Task 3

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


# Function to choose k using square root of the number of rows
def choose_k(dataframe):
    k = int(np.sqrt(dataframe.shape[0]))
    if k % 2 == 0:
        k += 1

    return k


# Load dataset
df = pd.read_csv("../data/H1.csv")

# List of columns to clean
wash_columns = ['ADR', 'BookingChanges', 'PreviousCancellations', 'PreviousBookingsNotCanceled', 'Adults', 'Children',
                'Babies', 'IsRepeatedGuest', 'RequiredCarParkingSpaces', 'TotalOfSpecialRequests']

# Replace zero values with NaN and fill NaN with mean
for col in wash_columns:
    df[col] = df[col].replace(0, np.nan)
    mean = df[col].mean(skipna=True)
    df[col] = df[col].replace(np.nan, mean)

# Define features (X) and target (y)
y = df['IsCanceled']
X = df[wash_columns]

# Split data into training and test sets with 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data as KNN is sensitive to the scale of the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Choose k using square root of the number of samples
k = choose_k(y_test)

# Train the model
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions on the test data
predicted_values = knn.predict(X_test)

# Accuracy score of the model (percentage of correct predictions)
accuracy = accuracy_score(y_test, predicted_values)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate confusion matrix
cm = confusion_matrix(y_test, predicted_values)
print("Confusion matrix:")
print(cm)
