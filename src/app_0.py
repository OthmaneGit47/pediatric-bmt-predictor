import streamlit as st
import pandas as pd
import numpy as np
import os

# Define categorical mappings
hla_match_mapping = {"10/10": 0, "9/10": 1, "8/10": 2, "7/10": 3}
donor_abo_mapping = {"0": 0, "A": 1, "B": -1, "AB": 2}

# Streamlit UI
st.title("Bone Marrow Transplant Success Prediction")

st.sidebar.header("Enter Patient Data")

# Collect input data
patient_data = {
    "Recipient Gender": st.sidebar.radio("Recipient Gender", ["Male", "Female"]),
    "Stem Cell Source": st.sidebar.radio("Stem Cell Source", ["Peripheral blood", "Bone marrow"]),
    "Donor Age": st.sidebar.slider("Donor Age", 0, 80, 30),
    "Donor Age <35": st.sidebar.radio("Donor Age <35", ["Yes", "No"]),
    "Acute GvHD Stage II-IV": st.sidebar.radio("Acute GvHD Stage II-IV", ["Yes", "No"]),
    "Gender Match": st.sidebar.radio("Gender Match", ["Female to Male", "Other"]),
    "Donor ABO": st.sidebar.selectbox("Donor ABO", list(donor_abo_mapping.keys())),
    "Recipient ABO": st.sidebar.selectbox("Recipient ABO", list(donor_abo_mapping.keys())),
    "Recipient Rh": st.sidebar.radio("Recipient Rh", ["+", "-"]),
    "ABO Match": st.sidebar.radio("ABO Match", ["Matched", "Mismatched"]),
    "CMV Status": st.sidebar.slider("CMV Status", 0, 10, 5),
    "Recipient CMV": st.sidebar.radio("Recipient CMV", ["Presence", "Absence"]),
    "Disease": st.sidebar.selectbox("Disease", ["ALL", "AML", "Chronic", "Nonmalignant", "Lymphoma"]),
    "Risk Group": st.sidebar.radio("Risk Group", ["High", "Low"]),
    "Second Transplant After Relapse": st.sidebar.radio("Second Transplant After Relapse", ["Yes", "No"]),
    "Disease Group": st.sidebar.radio("Disease Group", ["Malignant", "Nonmalignant"]),
    "HLA Match": st.sidebar.selectbox("HLA Match", list(hla_match_mapping.keys())),
    "HLA Mismatch": st.sidebar.radio("HLA Mismatch", ["Matched", "Mismatched"]),
    "Antigen Difference": st.sidebar.slider("Antigen Difference", -1, 3, 0),
    "Allele Difference": st.sidebar.slider("Allele Difference", -1, 4, 0),
    "HLA Group I": st.sidebar.slider("HLA Group I", 0, 5, 0),
    "Recipient Age": st.sidebar.slider("Recipient Age", 0, 100, 30),
    "Recipient Age <10": st.sidebar.radio("Recipient Age <10", ["Yes", "No"]),
    "Recipient Age Interval": st.sidebar.selectbox("Recipient Age Interval", ["(0,5]", "(5,10]", "(10,20]"]),
    "Relapse": st.sidebar.radio("Relapse", ["Yes", "No"]),
    "Acute GvHD Stage III-IV": st.sidebar.radio("Acute GvHD Stage III-IV", ["Yes", "No"]),
    "Chronic GvHD": st.sidebar.radio("Chronic GvHD", ["Yes", "No"]),
    "CD34+ Cells (10^6/kg)": st.sidebar.slider("CD34+ Cell Dose", 0, 500, 50),
    "CD3+/CD34+ Ratio": st.sidebar.slider("CD3+/CD34+ Ratio", 0.0, 10.0, 1.0),
    "CD3+ Cells (10^8/kg)": st.sidebar.slider("CD3+ Cell Dose", 0, 500, 50),
    "Recipient Body Mass": st.sidebar.slider("Recipient Body Mass", 0, 200, 70),
    "ANC Recovery Time": st.sidebar.slider("ANC Recovery Time", 0, 100, 20),
    "Platelet Recovery Time": st.sidebar.slider("Platelet Recovery Time", 0, 100, 20),
    "Time to Acute GvHD Stage III-IV": st.sidebar.slider("Time to Acute GvHD Stage III-IV", 0, 100, 20),
    "Survival Time (Days)": st.sidebar.slider("Survival Time", 0, 5000, 1000)
}

# Convert to DataFrame
input_df = pd.DataFrame([patient_data])

# Placeholder Prediction Section
if st.button("Predict"):
    st.write("### Predicted Probability of Death: **(Placeholder - Model Not Integrated Yet)**")
    st.write("### Feature Importance (SHAP Summary Plot)")
    st.image("shap_placeholder.png", caption="SHAP summary plot (to be integrated)")
    st.write("### Individual Prediction Explanation")
    st.image("shap_waterfall_placeholder.png", caption="SHAP waterfall plot (to be integrated)")

    # Feedback Section
    add_feedback = st.checkbox("Would you like to provide feedback?")

    if add_feedback:
        correct_prediction = st.radio("Was the prediction correct?", ["Yes", "No"])
        feedback_text = st.text_area("Additional Notes (optional)")

        if st.button("Submit Feedback"):
            feedback_file = "feedback.csv"
            feedback_data = pd.DataFrame([[*input_df.iloc[0], "Placeholder Prediction", correct_prediction, feedback_text]],
                                         columns=[*input_df.columns, "Predicted Risk", "Correct Prediction", "Feedback"])
            if os.path.exists(feedback_file):
                feedback_data.to_csv(feedback_file, mode="a", header=False, index=False)
            else:
                feedback_data.to_csv(feedback_file, mode="w", header=True, index=False)

            st.success("Feedback submitted! Thank you for your input.")
