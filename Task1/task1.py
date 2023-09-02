# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:58:39 2023

@author: 91846
"""

import pandas as pd_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataSet = pd_iris.read_csv("iris.csv")

# Display the first few rows of the dataset
print(dataSet.head())

# Get basic statistics of the dataset
print(dataSet.describe())

print(dataSet.isnull().sum())

# Visualize the data
label_encoder = LabelEncoder()
dataSet['Species'] = label_encoder.fit_transform(dataSet['Species'])

X = dataSet.drop('Species', axis=1)
y = dataSet['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Additional evaluation metrics
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

