import joblib
import pandas as pd

# Load the trained model
model = joblib.load('randomforest_model1.pkl')


# Function to predict survival status
def predict_survival_status(patient_data):
    # The feature names
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
    # convert to data frame
    patient_df = pd.DataFrame(patient_data, columns=feature_names)

    # Predict survival status
    prediction = model.predict(patient_df)
    prediction_proba = model.predict_proba(patient_df)

    return prediction, prediction_proba
