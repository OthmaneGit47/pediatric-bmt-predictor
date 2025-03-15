import joblib

# Load the trained model
model = joblib.load("src/lightgbm_model.pkl")

import pandas as pd

# Example new patient data (replace with real input)
new_data = pd.DataFrame({
    "age": [45],
    "graft_source": [1],
    "hla_match": [2],
    "disease_status": [0],
    "cmv_status": [1]
})

# Ensure feature columns match the training set
print("New data prepared for prediction:", new_data)