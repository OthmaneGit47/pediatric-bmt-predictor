from flask import Flask, request, render_template_string, flash
import pandas as pd
from model_for_app_2 import predict_survival_status  # Ensure this function is correctly imported

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Mapping dictionaries for specific fields
donor_abo_mapping = {"0": 0, "A": 1, "B": -1, "AB": 2}
hla_match_mapping = {"10/10": 0, "9/10": 1, "8/10": 2, "7/10": 3}

# Feature choices (form fields) in the same order as required by your model
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
    "CMV Status": "number",
    "Recipient CMV": ["Presence", "Absence"],
    "Disease": ["ALL", "AML", "Chronic", "Nonmalignant", "Lymphoma"],
    "Risk Group": ["High", "Low"],
    "Second Transplant After Relapse": ["Yes", "No"],
    "Disease Group": ["Malignant", "Nonmalignant"],
    "HLA Match": hla_match_mapping.keys(),
    "HLA Mismatch": ["Matched", "Mismatched"],
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


# Preprocessing: convert all form values to numeric types as expected by the model.
def preprocess_input(data):
    processed = {}
    # Iterate over each expected field
    for key in feature_choices.keys():
        val = data.get(key)
        # Convert binary fields ("Yes"/"No") to 1/0
        if val == "Yes":
            processed[key] = 1
        elif val == "No":
            processed[key] = 0
        else:
            # For number fields, try converting to float; if it fails, keep as string.
            try:
                processed[key] = float(val)
            except:
                processed[key] = val

    # Apply specific mappings for categorical fields:
    processed["Recipient Gender"] = 1 if processed["Recipient Gender"] == "Male" else 0
    processed["Stem Cell Source"] = 0 if processed["Stem Cell Source"] == "Peripheral blood" else 1
    processed["Donor ABO"] = donor_abo_mapping.get(processed["Donor ABO"], processed["Donor ABO"])
    processed["Recipient ABO"] = donor_abo_mapping.get(processed["Recipient ABO"], processed["Recipient ABO"])
    processed["Recipient Rh"] = 1 if processed["Recipient Rh"] == "+" else 0
    processed["ABO Match"] = 1 if processed["ABO Match"] == "Matched" else 0
    processed["Recipient CMV"] = 1 if processed["Recipient CMV"] == "Presence" else 0
    # Map "Disease" to numeric codes (adjust these codes if needed)
    disease_mapping = {"ALL": 0, "AML": 1, "Chronic": 2, "Nonmalignant": 3, "Lymphoma": 4}
    processed["Disease"] = disease_mapping.get(processed["Disease"], processed["Disease"])
    # For "Risk Group" and "Disease Group"
    processed["Risk Group"] = 1 if processed["Risk Group"] == "High" else 0
    processed["Disease Group"] = 1 if processed["Disease Group"] == "Malignant" else 0
    # For "Gender Match"
    processed["Gender Match"] = 1 if processed["Gender Match"] == "Female to Male" else 0
    # For "HLA Match"
    processed["HLA Match"] = hla_match_mapping.get(processed["HLA Match"], processed["HLA Match"])
    # For "HLA Mismatch"
    processed["HLA Mismatch"] = 1 if processed["HLA Mismatch"] == "Matched" else 0
    # For "Recipient Age Interval"
    interval_mapping = {"(0,5]": 0, "(5,10]": 1, "(10,20]": 2}
    processed["Recipient Age Interval"] = interval_mapping.get(processed["Recipient Age Interval"],
                                                               processed["Recipient Age Interval"])

    # Return the data as a DataFrame (the order of columns should match what the model expects)
    return pd.DataFrame([processed])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data using the keys from feature_choices
        patient_data = {key: request.form.get(key) for key in feature_choices}
        # Preprocess the data so that all values are numeric
        input_df = preprocess_input(patient_data)
        # Predict using your model function
        prediction, prediction_proba = predict_survival_status(input_df)
        # Flash the prediction results
        flash(f"Predicted Survival Status: {prediction[0]}", "info")
        flash(f"Prediction Probability: {prediction_proba[0]}", "info")
        # Render the template with results
        return render_template_string(template,
                                      patient_data=patient_data,
                                      input_df=input_df.to_dict(orient='records')[0],
                                      feature_choices=feature_choices)
    return render_template_string(template, feature_choices=feature_choices)


# HTML Template
template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BMT Success Prediction</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {background-color: #f5f7fa;}
        .main-title {text-align: center; color: #00274D; font-size: 36px; font-weight: bold; margin-top: 20px;}
        .sidebar {background-color: #eef2f7; padding: 20px; border-radius: 10px; margin-top: 20px;}
        .form-group label {font-weight: bold; margin-top: 10px;}
        .btn-primary {width: 100%; margin-top: 15px;}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="main-title"><i class="fas fa-hospital-user"></i> BMT Success Prediction</h1>
        <div class="row">
            <div class="col-md-4 sidebar">
                <form method="POST">
                    <h4><i class="fas fa-user-md"></i> Enter Patient Data</h4>
                    {% for key, value in feature_choices.items() %}
                    <div class="form-group">
                        <label for="{{ key }}"><i class="fas fa-info-circle" title="Enter {{ key }}"></i> {{ key }}</label>
                        {% if value == "number" %}
                            <input type="number" class="form-control" id="{{ key }}" name="{{ key }}">
                        {% else %}
                            <select class="form-control" id="{{ key }}" name="{{ key }}">
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
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
