import joblib
import numpy as np

# Load the pre-trained model
model = joblib.load('randomforest_model1.pkl')

# Define the features (columns) expected by the model
# These should match the features used during training
FEATURES = [
    'Recipientgender', 'Stemcellsource', 'Donorage', 'Donorage35', 'IIIV', 
    'Gendermatch', 'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 
    'CMVstatus', 'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 
    'Txpostrelapse', 'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Antigen', 
    'Allele', 'Recipientage10', 'Recipientageint', 'Relapse', 'aGvHDIIIIV', 
    'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass', 
    'ANCrecovery', 'PLTrecovery', 'time_to_aGvHD_III_IV', 'survival_time'
]

# Example input for a single patient
# Replace these values with the actual patient's data
patient_data = {
    'Recipientgender': 1.0,
    'Stemcellsource': 1.0,
    'Donorage': 21.128767,
    'Donorage35': 0.0,
    'IIIV': 1.0,
    'Gendermatch': 0.0,
    'DonorABO': 0.0,
    'RecipientABO': 0.0,
    'RecipientRh': 1.0,
    'ABOmatch': 0.0,
    'CMVstatus': 0.0,
    'DonorCMV': 0.0,
    'RecipientCMV': 0.0,
    'Disease': 0.0,
    'Riskgroup': 1.0,
    'Txpostrelapse': 0.0,
    'Diseasegroup': 1.0,
    'HLAmatch': 0.0,
    'HLAmismatch': 0.0,
    'Antigen': -1.0,
    'Allele': -1.0,
    'Recipientage10': 1.0,
    'Recipientageint': 2.0,
    'Relapse': 0.0,
    'aGvHDIIIIV': 1.0,
    'extcGvHD': 1.0,
    'CD34kgx10d6': 5.08,
    'CD3dCD34': 0.709805,
    'CD3dkgx10d8': 7.16,
    'Rbodymass': 54.1,
    'ANCrecovery': 15.0,
    'PLTrecovery': 17.0,
    'time_to_aGvHD_III_IV': 1000000.0,
    'survival_time': 20.0
}

# Convert the patient data into a numpy array
input_data = np.array([patient_data[feature] for feature in FEATURES]).reshape(1, -1)

# Make a prediction
prediction = model.predict(input_data)  # Binary prediction (0 or 1)

# Output the result
if prediction[0] == 0:
    print("Prediction: Live (Success)")
else:
    print("Prediction: Die (Failure)")