from flask import Flask, render_template, request, abort
import lightgbm as lgb
import pandas as pd
import pickle
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
from io import BytesIO

app = Flask(__name__)

# Load the trained model safely
model_path = 'model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print("Warning: model.pkl not found. Predictions will not work.")

# Load feature names safely
data_path = 'processed_data_v3.csv'
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    if 'survival_status' in df.columns:
        feature_names = df.drop('survival_status', axis=1).columns.tolist()
    else:
        feature_names = df.columns.tolist()
else:
    feature_names = []
    print("Warning: processed_data_v3.csv not found. Feature names unavailable.")

# Function to generate SHAP explanation
def generate_shap_plot(input_df):
    if model is None:
        return None  # Avoid errors if model is missing
    
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    
    return encoded_img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    shap_img = None
    result = None

    if request.method == 'POST' and model:
        try:
            # Extract input values from form
            input_data = [float(request.form[feature]) for feature in feature_names]
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Predict survival status
            prediction = model.predict(input_df)[0]
            result = "Success" if prediction == 1 else "Failure"
            
            # Generate SHAP explanation
            shap_img = generate_shap_plot(input_df)
        except Exception as e:
            result = f"Error: {str(e)}"

    # Check if index.html exists before rendering
    if not os.path.exists('templates/index.html'):
        return "Error: Missing index.html file in the templates folder.", 500
    
    return render_template('index.html', feature_names=feature_names, prediction=result, shap_img=shap_img)

if __name__ == '__main__':
    app.run(debug=True)
