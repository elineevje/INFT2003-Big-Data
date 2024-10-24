# Exercise 5, task 1

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('../data/titanic.csv')

# Convert sex-column to numeric values
cleanup_nums = {"Sex": {"male": 1, "female": 0}}
df.replace(cleanup_nums, inplace=True)

# Drop rows where age is missing
df = df[df['Age'].notnull()]

print(df.head())

# Define target (survived) and features (all columns except survived)
y = df['Survived']
X = df.drop('Survived', axis=1)
print(X.head())

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a decision tree classifier
classifier = DecisionTreeClassifier(min_impurity_decrease=0.01, max_depth=5)
# Fit the classifier by using the training data
classifier.fit(X_train, y_train)

# Predict by using the test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier by comparing the predicted values with the actual values
# Value 0 is Not Survived and value 1 is Survived
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))


# Delete old dotfile before creating a new one
if os.path.exists("../tree/dtc.dot"):
    os.remove("../tree/dtc.dot")

# Export the decision tree to a dot file
dotfile = open("../tree/dtc.dot", 'w')
tree.export_graphviz(classifier, out_file=dotfile, feature_names=X.columns, class_names=['Not Survived', 'Survived'])
dotfile.close()

