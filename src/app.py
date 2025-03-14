import streamlit as st
import pandas as pd
import os

# Encode categorical variables (example mappings)
donor_abo_mapping = {"A": 0, "B": 1, "AB": 2, "O": 3}
hla_match_mapping = {"Full Match": 0, "Partial Match": 1, "Mismatched": 2}

# Streamlit UI
st.title("Bone Marrow Transplant Success Prediction")

st.sidebar.header("Enter Patient Data")
patient_data = {
    "Recipient Gender": st.sidebar.radio("Recipient Gender", ["Male", "Female"]),
    "Donor Age": st.sidebar.slider("Donor Age", 0, 80, 30),
    "HLA Match": st.sidebar.selectbox("HLA Match", list(hla_match_mapping.keys())),
    "Donor ABO": st.sidebar.selectbox("Donor ABO", list(donor_abo_mapping.keys())),
    "Recipient ABO": st.sidebar.selectbox("Recipient ABO", list(donor_abo_mapping.keys())),
    "CD34+ Cells (10^6/kg)": st.sidebar.slider("CD34+ Cell Dose", 0, 500, 50),
    "Survival Time (Days)": st.sidebar.slider("Survival Time", 0, 5000, 1000),
}

# Convert to DataFrame
input_df = pd.DataFrame([patient_data])

# Placeholder Prediction Section
if st.button("Predict"):
    st.write("### Predicted Probability of Death: **(Placeholder - Model Not Integrated Yet)**")

    # Placeholder SHAP Explanation
    st.write("### Feature Importance (SHAP Summary Plot)")
    st.image("shap_placeholder.png", caption="SHAP summary plot (to be integrated)")

    st.write("### Individual Prediction Explanation")
    st.image("shap_waterfall_placeholder.png", caption="SHAP waterfall plot (to be integrated)")

    #  OPTIONAL FEEDBACK SECTION
    add_feedback = st.checkbox("Would you like to provide feedback?")

    if add_feedback:
        correct_prediction = st.radio("Was the prediction correct?", ["Yes", "No"])
        feedback_text = st.text_area("Additional Notes (optional)")

        if st.button("Submit Feedback"):
            feedback_data = pd.DataFrame(
                [[
                    input_df.iloc[0]["Recipient Gender"], input_df.iloc[0]["Donor Age"],
                    input_df.iloc[0]["HLA Match"], input_df.iloc[0]["Donor ABO"],
                    input_df.iloc[0]["Recipient ABO"], input_df.iloc[0]["CD34+ Cells (10^6/kg)"],
                    input_df.iloc[0]["Survival Time (Days)"], "Placeholder Prediction",
                    correct_prediction, feedback_text
                ]],
                columns=["Recipient Gender", "Donor Age", "HLA Match", "Donor ABO", "Recipient ABO",
                        "CD34+ Cells (10^6/kg)", "Survival Time (Days)", "Predicted Risk",
                        "Correct Prediction", "Feedback"]
            )

            # Append to CSV
            feedback_file = "feedback.csv"
            if os.path.exists(feedback_file):
                feedback_data.to_csv(feedback_file, mode="a", header=False, index=False)
            else:
                feedback_data.to_csv(feedback_file, mode="w", header=True, index=False)

            st.success(" Feedback submitted! Thank you for your input.")
