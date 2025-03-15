from flask import Flask, request, render_template_string, flash, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

donor_abo_mapping = {"0": 0, "A": 1, "B": -1, "AB": 2}
hla_match_mapping = {"10/10": 0, "9/10": 1, "8/10": 2, "7/10": 3}

def preprocess_input(data):
    data["Recipient Gender"] = 1 if data["Recipient Gender"] == "Male" else 0
    data["Donor ABO"] = donor_abo_mapping[data["Donor ABO"]]
    data["Recipient ABO"] = donor_abo_mapping[data["Recipient ABO"]]
    data["HLA Match"] = hla_match_mapping[data["HLA Match"]]
    return pd.DataFrame([data])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        patient_data = {key: request.form.get(key) for key in request.form}
        input_df = preprocess_input(patient_data)
        flash("Predicted Probability of Death: (Placeholder - Model Not Integrated Yet)", "info")
        return render_template_string(template, patient_data=patient_data, input_df=input_df.to_dict(orient='records')[0], donor_abo_mapping=donor_abo_mapping, hla_match_mapping=hla_match_mapping)
    return render_template_string(template, donor_abo_mapping=donor_abo_mapping, hla_match_mapping=hla_match_mapping)

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
                    {% for key, value in {
                        "Recipient Gender": ["Male", "Female"],
                        "Stem Cell Source": ["Peripheral blood", "Bone marrow"],
                        "Donor Age": "number",
                        "Donor Age <35": ["Yes", "No"],
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
                        "Recipient Age <10": ["Yes", "No"],
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
                    }.items() %}
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
            </div>
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)