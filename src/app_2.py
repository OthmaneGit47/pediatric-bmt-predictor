import joblib
import shap
import base64
from io import BytesIO
from flask import Flask, request, render_template_string, flash
import pandas as pd
from model_for_app_2 import predict_survival_status  # Ensure this function is correctly imported

# Load the trained model
model = joblib.load('randomforest_model1.pkl')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Mapping dictionaries for specific fields
donor_abo_mapping = {"0": 0, "A": 1, "B": -1, "AB": 2}
hla_match_mapping = {"10/10": 0, "9/10": 1, "8/10": 2, "7/10": 3}

# Feature choices (form fields) as shown to the user.
feature_choices = {
    "Recipient Gender": ["Male", "Female"],
    "Stem Cell Source": ["Peripheral blood", "Bone marrow"],
    "Donor Age": "number",
    "Acute GvHD Stage II-IV": ["Yes", "No"],
    "Gender Match": ["Female to Male", "Other"],
    "Donor ABO": donor_abo_mapping.keys(),
    "Recipient ABO": donor_abo_mapping.keys(),
    "Recipient Rh": ["+", "-"],
    "ABO Match": ["Matched", "Mismatched"],
    "CMV Status": "number",  # Expect numeric input: e.g., 1 or 3.
    "Recipient CMV": ["Presence", "Absence"],
    "Disease": ["ALL", "AML", "Chronic", "Nonmalignant", "Lymphoma"],
    "Risk Group": ["High", "Low"],
    "Second Transplant After Relapse": ["Yes", "No"],
    "Disease Group": ["Malignant", "Nonmalignant"],
    "HLA Match": hla_match_mapping.keys(),
    "Antigen Difference": "number",
    "Allele Difference": "number",
    "HLA Group I": "number",
    "Recipient Age": "number",
    "Recipient Age Interval": ["(0,5]", "(5,10]", "(10,20]"],
    "Relapse": ["Yes", "No"],
    "Acute GvHD Stage III-IV": ["Yes", "No"],
    "Chronic GvHD": ["Yes", "No"],
    "CD34+ Cells (10^6/kg)": "number",
    "CD3+/CD34+ Ratio": "number",
    "CD3+ Cells (10^8/kg)": "number",
    "Recipient Body Mass": "number",
    "ANC Recovery Time": "number",
    "Platelet Recovery Time": "number",
    "Time to Acute GvHD Stage III-IV": "number",
    "Survival Time (Days)": "number"
}

def preprocess_input(data):
    # (Preprocessing code as before, unchanged.)
    intermediate = {}
    intermediate["Recipientgender"] = 1 if data.get("Recipient Gender") == "Male" else 0
    intermediate["Stemcellsource"] = 0 if data.get("Stem Cell Source") == "Peripheral blood" else 1
    intermediate["Donorage"] = float(data.get("Donor Age"))
    intermediate["IIIV"] = 1 if data.get("Acute GvHD Stage II-IV") == "Yes" else 0
    intermediate["Gendermatch"] = 1 if data.get("Gender Match") == "Female to Male" else 0
    intermediate["RecipientRh"] = 1 if data.get("Recipient Rh") == "+" else 0
    intermediate["ABOmatch"] = 1 if data.get("ABO Match") == "Matched" else 0
    try:
        cmv_val = float(data.get("CMV Status"))
    except:
        cmv_val = 0
    intermediate["CMVstatus_1.0"] = 1 if cmv_val == 1 else 0
    intermediate["CMVstatus_3.0"] = 1 if cmv_val == 3 else 0
    intermediate["RecipientCMV"] = 1 if data.get("Recipient CMV") == "Presence" else 0
    intermediate["Riskgroup"] = 1 if data.get("Risk Group") == "High" else 0
    intermediate["Txpostrelapse"] = 1 if data.get("Second Transplant After Relapse") == "Yes" else 0
    intermediate["Diseasegroup"] = 1 if data.get("Disease Group") == "Malignant" else 0
    intermediate["HLAmismatch"] = hla_match_mapping.get(data.get("HLA Match"), 0)
    intermediate["Relapse"] = 1 if data.get("Relapse") == "Yes" else 0
    intermediate["aGvHDIIIIV"] = 1 if data.get("Acute GvHD Stage III-IV") == "Yes" else 0
    intermediate["extcGvHD"] = 1 if data.get("Chronic GvHD") == "Yes" else 0
    intermediate["CD34kgx10d6"] = float(data.get("CD34+ Cells (10^6/kg)"))
    intermediate["CD3dCD34"] = float(data.get("CD3+/CD34+ Ratio"))
    intermediate["CD3dkgx10d8"] = float(data.get("CD3+ Cells (10^8/kg)"))
    intermediate["RecipientBodyMass"] = float(data.get("Recipient Body Mass")) if data.get("Recipient Body Mass") else 0
    intermediate["ANCrecovery"] = float(data.get("ANC Recovery Time"))
    intermediate["PLTrecovery"] = float(data.get("Platelet Recovery Time"))
    intermediate["time_to_aGvHD_III_IV"] = float(data.get("Time to Acute GvHD Stage III-IV"))
    intermediate["survival_time"] = float(data.get("Survival Time (Days)"))
    for col in ["DonorABO_-1", "DonorABO_0", "DonorABO_1", "DonorABO_2"]:
        intermediate[col] = 0
    donor_val = donor_abo_mapping.get(data.get("Donor ABO"))
    if donor_val is not None:
        if donor_val == -1:
            intermediate["DonorABO_-1"] = 1
        elif donor_val == 0:
            intermediate["DonorABO_0"] = 1
        elif donor_val == 1:
            intermediate["DonorABO_1"] = 1
        elif donor_val == 2:
            intermediate["DonorABO_2"] = 1
    for col in ["RecipientABO_-1.0", "RecipientABO_0.0", "RecipientABO_1.0", "RecipientABO_2.0"]:
        intermediate[col] = 0
    recip_val = donor_abo_mapping.get(data.get("Recipient ABO"))
    if recip_val is not None:
        if recip_val == -1:
            intermediate["RecipientABO_-1.0"] = 1
        elif recip_val == 0:
            intermediate["RecipientABO_0.0"] = 1
        elif recip_val == 1:
            intermediate["RecipientABO_1.0"] = 1
        elif recip_val == 2:
            intermediate["RecipientABO_2.0"] = 1
    for col in ["Disease_ALL", "Disease_AML", "Disease_chronic", "Disease_lymphoma", "Disease_nonmalignant"]:
        intermediate[col] = 0
    disease = data.get("Disease")
    if disease == "ALL":
        intermediate["Disease_ALL"] = 1
    elif disease == "AML":
        intermediate["Disease_AML"] = 1
    elif disease == "Chronic":
        intermediate["Disease_chronic"] = 1
    elif disease == "Lymphoma":
        intermediate["Disease_lymphoma"] = 1
    elif disease == "Nonmalignant":
        intermediate["Disease_nonmalignant"] = 1
    for col in ["Antigen_0.0", "Antigen_2.0"]:
        intermediate[col] = 0
    antigen_diff = float(data.get("Antigen Difference"))
    if antigen_diff == 0.0:
        intermediate["Antigen_0.0"] = 1
    elif antigen_diff == 2.0:
        intermediate["Antigen_2.0"] = 1
    for col in ["Allele_0.0", "Allele_1.0", "Allele_3.0"]:
        intermediate[col] = 0
    allele_diff = float(data.get("Allele Difference"))
    if allele_diff == 0.0:
        intermediate["Allele_0.0"] = 1
    elif allele_diff == 1.0:
        intermediate["Allele_1.0"] = 1
    elif allele_diff == 3.0:
        intermediate["Allele_3.0"] = 1
    for col in ["HLAgrI_3", "HLAgrI_4", "HLAgrI_5", "HLAgrI_7"]:
        intermediate[col] = 0
    hla_group = float(data.get("HLA Group I"))
    if hla_group == 3:
        intermediate["HLAgrI_3"] = 1
    elif hla_group == 4:
        intermediate["HLAgrI_4"] = 1
    elif hla_group == 5:
        intermediate["HLAgrI_5"] = 1
    elif hla_group == 7:
        intermediate["HLAgrI_7"] = 1
    age_interval = data.get("Recipient Age Interval")
    if age_interval == "(0,5]":
        intermediate["Recipientageint_0"] = 1
        intermediate["Recipientageint_1"] = 0
    elif age_interval == "(5,10]":
        intermediate["Recipientageint_0"] = 0
        intermediate["Recipientageint_1"] = 1
    else:
        intermediate["Recipientageint_0"] = 0
        intermediate["Recipientageint_1"] = 0
    training_features = [
        'Recipientgender', 'Stemcellsource', 'Donorage', 'IIIV', 'Gendermatch', 'RecipientRh', 'ABOmatch',
        'CMVstatus_1.0', 'CMVstatus_3.0', 'RecipientCMV', 'Riskgroup', 'Txpostrelapse', 'Diseasegroup', 'HLAmismatch', 'Relapse',
        'aGvHDIIIIV', 'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'ANCrecovery', 'PLTrecovery',
        'time_to_aGvHD_III_IV', 'survival_time', 'DonorABO_-1', 'DonorABO_0', 'DonorABO_1', 'DonorABO_2',
        'RecipientABO_-1.0', 'RecipientABO_0.0', 'RecipientABO_1.0', 'RecipientABO_2.0',
        'Disease_ALL', 'Disease_AML', 'Disease_chronic', 'Disease_lymphoma', 'Disease_nonmalignant',
        'Antigen_0.0', 'Antigen_2.0', 'Allele_0.0', 'Allele_1.0', 'Allele_3.0', 'HLAgrI_3', 'HLAgrI_4', 'HLAgrI_5',
        'HLAgrI_7', 'Recipientageint_0', 'Recipientageint_1'
    ]
    for feat in training_features:
        if feat not in intermediate:
            intermediate[feat] = 0
    df = pd.DataFrame([intermediate])
    return df[training_features]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        patient_data = {key: request.form.get(key) for key in feature_choices}
        missing_fields = [key for key, value in patient_data.items() if not value]
        if missing_fields:
            flash("Please fill in all fields before submitting.", "danger")
            return render_template_string(template,
                                          patient_data=patient_data,
                                          feature_choices=feature_choices,
                                          missing_fields=missing_fields)
        input_df = preprocess_input(patient_data)
        prediction, prediction_proba = predict_survival_status(input_df)
        # Initialize the SHAP explainer with a background dataset.
        explainer = shap.TreeExplainer(model, data=input_df, feature_perturbation='interventional')
        shap_values = explainer.shap_values(input_df, check_additivity=False)
        shap_vector = shap_values[0][0]
        features_list = list(input_df.columns)
        if len(shap_vector) != len(features_list):
            min_len = min(len(shap_vector), len(features_list))
            shap_vector = shap_vector[:min_len]
            features_list = features_list[:min_len]
        # Generate HTML force plot.
        shap_html = shap.force_plot(
            explainer.expected_value[0],
            shap_vector,
            feature_names=features_list,
            show=False
        ).html()
        # Call shap.initjs() to get the necessary JS library snippet.
        shap_init = shap.initjs()
        flash(f"Predicted Survival Status: {prediction[0]}", "info")
        return render_template_string(template,
                                      patient_data=patient_data,
                                      feature_choices=feature_choices,
                                      missing_fields=[],
                                      shap_html=shap_html,
                                      shap_init=shap_init)
    return render_template_string(template, feature_choices=feature_choices, missing_fields=[])

template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMT Success Prediction</title>
    {{ shap_init | safe }}
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {background-color: #f5f7fa;}
        .main-title {text-align: center; color: #00274D; font-size: 36px; font-weight: bold; margin-top: 20px;}
        .sidebar {background-color: #eef2f7; padding: 20px; border-radius: 10px; margin-top: 20px;}
        .form-group label {font-weight: bold; margin-top: 10px;}
        .btn-primary {width: 100%; margin-top: 15px;}
        .missing-field {border: 2px solid red !important;}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="main-title"><i class="fas fa-hospital-user"></i> BMT Success Prediction</h1>
        <div class="row">
            <div class="col-md-4 sidebar">
                <form method="POST" id="predictionForm">
                    <h4><i class="fas fa-user-md"></i> Enter Patient Data</h4>
                    {% for key, value in feature_choices.items() %}
                    <div class="form-group">
                        <label for="{{ key }}"><i class="fas fa-info-circle" title="Enter {{ key }}"></i> {{ key }}</label>
                        {% if value == "number" %}
                            <input type="number" class="form-control" id="{{ key }}" name="{{ key }}">
                        {% else %}
                            <select class="form-control" id="{{ key }}" name="{{ key }}">
                                <option value="">-- Select --</option>
                                {% for option in value %}
                                    <option>{{ option }}</option>
                                {% endfor %}
                            </select>
                        {% endif %}
                    </div>
                    {% endfor %}
                    <button type="submit" class="btn btn-primary"><i class="fas fa-diagnoses"></i> Predict</button>
                </form>
                <br>
                {% with messages = get_flashed_messages(with_categories=true) %}
                    {% if messages %}
                        {% for category, message in messages %}
                            <div class="alert alert-info">{{ message }}</div>
                        {% endfor %}
                    {% endif %}
                {% endwith %}
            </div>
            {% if shap_html %}
            <div class="col-md-8">
                <h4>SHAP Explanation for Prediction</h4>
                <div id="force_plot">
                    {{ shap_html | safe }}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    <script>
        document.getElementById("predictionForm").addEventListener("submit", function(event) {
            let isValid = true;
            document.querySelectorAll(".form-control").forEach(function(element) {
                if (!element.value.trim()) {
                    element.classList.add("missing-field");
                    isValid = false;
                } else {
                    element.classList.remove("missing-field");
                }
            });
            if (!isValid) {
                event.preventDefault();
                alert("Please fill in all required fields before submitting.");
            }
        });
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
