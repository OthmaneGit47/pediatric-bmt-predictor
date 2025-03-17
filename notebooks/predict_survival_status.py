import joblib
import pandas as pd
import numpy as np
import os

# Load the trained model
print(os.path.exists("randomforest_model1.pkl")) 
rf_model = joblib.load('src\lightgbm_model.pkl')

# Example of new patient data
new_patient_data = np.array([[1.0, 1.0, 42.380822, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 6.34, 1.231287, 5.15, 15.0, 18.0, 1000000.0, 330.0, True, False, False, False, False, False, False, True, False, False, False, False, False, False, True, True, False, False, True, False, False, False, False, False, False, True]])

# Define the feature names
feature_names = [
    'Recipientgender', 'Stemcellsource', 'Donorage', 'IIIV', 'Gendermatch', 'RecipientRh', 'ABOmatch',
    'DonorCMV', 'RecipientCMV', 'Riskgroup', 'Txpostrelapse', 'Diseasegroup', 'HLAmismatch', 'Relapse',
    'aGvHDIIIIV', 'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'ANCrecovery', 'PLTrecovery',
    'time_to_aGvHD_III_IV', 'survival_time', 'DonorABO_-1', 'DonorABO_0', 'DonorABO_1', 'DonorABO_2',
    'RecipientABO_-1.0', 'RecipientABO_0.0', 'RecipientABO_1.0', 'RecipientABO_2.0', 'CMVstatus_1.0',
    'CMVstatus_3.0', 'Disease_ALL', 'Disease_AML', 'Disease_chronic', 'Disease_lymphoma', 'Disease_nonmalignant',
    'Antigen_0.0', 'Antigen_2.0', 'Allele_0.0', 'Allele_1.0', 'Allele_3.0', 'HLAgrI_3', 'HLAgrI_4', 'HLAgrI_5',
    'HLAgrI_7', 'Recipientageint_0', 'Recipientageint_1'
]


# Convert the new patient data into a DataFrame
new_patient_df = pd.DataFrame(new_patient_data, columns=feature_names)

# Make the prediction
prediction = rf_model.predict(new_patient_df)
prediction_proba = rf_model.predict_proba(new_patient_df)  # Get probabilities if needed

# Print the results
print("Predicted Survival Status:", prediction[0])
print("Prediction Probability:", prediction_proba)
