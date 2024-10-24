# Exercise 5, task 3

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('../data/boston-housing-reduced.csv')

# Replace '?' with NaN and fill NaN with the mean
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Define target (MEDV) and features (all columns except MEDV)
y = df['MEDV']
X = df.drop('MEDV', axis=1)

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a decision tree regressor
dtr = DecisionTreeRegressor(min_impurity_decrease=1, max_depth=5)

# Fit the regressor by using the training data
dtr.fit(X_train, y_train)

# Predict by using the test data
y_pred = dtr.predict(X_test)

# Evaluate the regressor by printing the training and test scores
print("Training score: " + str(dtr.score(X_train, y_train)))
print("Test score: " + str(dtr.score(X_test, y_test)))

# Delete old dotfile before creating a new one
if os.path.exists("../tree/dtr.dot"):
    os.remove("../tree/dtr.dot")

# Export the decision tree to a dot file
dotfile = open("../tree/dtr.dot", 'w')
tree.export_graphviz(dtr, out_file=dotfile, feature_names=X.columns)
dotfile.close()
