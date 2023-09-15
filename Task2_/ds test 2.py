# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:32:28 2023

@author: 91846
"""

import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind  # Import the ttest_ind function from scipy.stats

# Specify the path to the ZIP file containing the dataset
zip_file_path = "C:\\Users\\91846\\Downloads\\archive (3).zip"

# Specify the name of the CSV file within the ZIP archive (use the correct name)
csv_file_name = "Unemployment_Rate_upto_11_2020.csv"

# Extract the CSV file from the ZIP archive and load it into a DataFrame
with zipfile.ZipFile(zip_file_path, "r") as zip_file:
    with zip_file.open(csv_file_name) as csv_file:
        unemployment_data = pd.read_csv(csv_file)

# Explore the dataset
print(unemployment_data.head())
print(unemployment_data.columns)
  # Display the first few rows of the dataset
plt.figure(figsize=(10, 6))
plt.plot(unemployment_data[' Date'], unemployment_data[' Estimated Unemployment Rate (%)'], marker='o', linestyle='-', color='b')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# Hypothesis Testing
# Example: Test whether there is a significant difference in unemployment rate before and during COVID-19
covid_start_date = "2020-03-01"
before_covid = unemployment_data[unemployment_data[' Date'] < covid_start_date][' Estimated Unemployment Rate (%)']
during_covid = unemployment_data[unemployment_data[' Date'] >= covid_start_date][' Estimated Unemployment Rate (%)']

# Perform a t-test
t_stat, p_value = ttest_ind(before_covid, during_covid)

# Print the results
if p_value < 0.05:
    print("There is a significant difference in unemployment rate before and during COVID-19.")
else:
    print("There is no significant difference in unemployment rate before and during COVID-19.")

# Conclusion and Recommendations
# You can summarize your findings and provide recommendations here.
# Conclusion and Recommendations
# You can summarize your findings and provide recommendations based on your analysis.

# Print the t-statistic and p-value
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Interpret the results
if p_value < 0.05:
    print("Conclusion: There is a significant difference in unemployment rate before and during COVID-19.")
    print("Recommendation: Further analyze the factors contributing to this difference and consider policy interventions.")
else:
    print("Conclusion: There is no significant difference in unemployment rate before and during COVID-19.")
    print("Recommendation: Continue monitoring and analyzing unemployment trends for informed decision-making.")

# You can further elaborate on your analysis and provide additional insights and recommendations as needed.


