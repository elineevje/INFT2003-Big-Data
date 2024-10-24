# Exercise 5, task 2

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('../data/ov4-breast-cancer.csv')

# Replace '?' with NaN and fill NaN with the mean
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

# Define target (classes) and features (all columns except classes)
y = df['classes']
X = df.drop('classes', axis=1)

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a decision tree classifier
dtc = DecisionTreeClassifier(min_impurity_decrease=0.01, max_depth=3)

# Fit the classifier by using the training data
dtc.fit(X_train, y_train)

# Predict by using the test data
y_pred = dtc.predict(X_test)

# Evaluate the classifier by comparing the predicted values with the actual values
# Value 0 is Benign and value 1 is Malignant
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# Delete old dotfile before creating a new one
if os.path.exists("../tree/dtc2.dot"):
    os.remove("../tree/dtc2.dot")

# Export the decision tree to a dot file
dotfile = open("../tree/dtc2.dot", 'w')
tree.export_graphviz(dtc, out_file=dotfile, feature_names=X.columns, class_names=['Benign', 'Malignant'])
dotfile.close()



