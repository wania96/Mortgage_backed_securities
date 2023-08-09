#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
with open(r"C:\Users\wania_96\Desktop\LoanExport\classification_.pkl", 'rb') as file:
    classification_pipeline = pickle.load(file)

with open(r"C:\Users\wania_96\Desktop\LoanExport\regression_.pkl", 'rb') as file:
    regression_pipeline = pickle.load(file)    


# Function to collect user input for classification and regression
def get_input():
    # Input for common features
    common_features = ["DTI", "CreditRange"]

    # Input for classification
    classification_columns = ["MSA", "MIP", "OCLTV", "OrigUPB", "PropertyState", "MonthsDelinquent", "LTV_Range", "RepPayRange"]

    # Input for regression
    regression_columns = ["PPM", "NumBorrowers", "ServicerName", "EverDelinquent", "IsFirstTimeHomebuyer", "monthly_income"]

    common_input = [input(f"Enter {feature}: ") for feature in common_features]

    # Input for classification
    classification_input = [input(f"Enter {col}: ") for col in classification_columns]

    # Input for regression
    regression_input = [input(f"Enter {col}: ") for col in regression_columns]

    # Create DataFrames for classification and regression inputs separately
    data_classification = pd.DataFrame([common_input + classification_input], columns=common_features + classification_columns)
    data_regression = pd.DataFrame([common_input + regression_input], columns=common_features + regression_columns)

    return data_classification, data_regression

# Get user input for common features and for classification and regression
data_classification, data_regression = get_input()

# Use the loaded classification pipeline to make predictions on the user input data
classification_predictions = classification_pipeline.predict(data_classification)

# Use the loaded regression pipeline to make predictions on the user input data
regression_predictions = regression_pipeline.predict(data_regression)

# Convert the predictions to 1-dimensional arrays
classification_predictions = classification_predictions.flatten()
regression_predictions = regression_predictions.flatten()

# Combine the predictions from both models
combined_predictions = pd.DataFrame(data={"Classification_Prediction_Defaulter": classification_predictions, "Regression_Prediction_prepayment": regression_predictions})

# Display the final combined predictions
print(combined_predictions)


# In[2]:


# Save the combined predictions as a pickle file
output_file_path = r"C:\Users\wania_96\Desktop\LoanExport\combined_predictions.pkl"
with open(output_file_path, 'wb') as f:
    pickle.dump(combined_predictions, f)


# In[ ]:




