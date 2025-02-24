import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load Trained Model & Label Encoder
model_path = "models/xgboost_model.pkl"
encoder_path = "models/label_encoder.pkl"
data_url = "https://raw.githubusercontent.com/nidhi0805/HarmReduction-Project/refs/heads/main/Data/Processed/cleaned_drug_testing_data.csv"  # Load the dataset for naloxone response analysis

if os.path.exists(model_path) and os.path.exists(encoder_path):
    model = joblib.load(model_path)
    le = joblib.load(encoder_path)
else:
    st.error("🚨 Model files not found! Ensure 'models/' contains 'xgboost_model.pkl' & 'label_encoder.pkl'")
    st.stop()

# Load Dataset to Analyze Naloxone Response
df = pd.read_csv(data_url)

# Get list of drugs from Label Encoder
drug_list = sorted(le.classes_.tolist())

# Streamlit UI
st.title("💊 Drug Overdose Risk Prediction")
st.write("Select a drug to predict **complex overdose risk** and **naloxone effectiveness**.")

# Dropdown for Drug Selection
drug_name = st.selectbox("🔹 Choose a Drug", drug_list)

def get_naloxone_effectiveness(drug_name):
    # Normalize Drug Name for Matching
    drug_df = df[df["Sold as"].str.strip().str.lower() == drug_name.strip().lower()]
    
    print(f"🔍 Checking Drug: {drug_name}")
    print(f"🔹 Total Rows Found: {len(drug_df)}")
    
    if len(drug_df) == 0:
        return 0.0  # No data found for this drug
    
    print(drug_df[["Sold as", "Naloxone_Response"]].head())  # Debugging
    
    # Ensure Naloxone Response is Numeric
    drug_df["Naloxone_Response"] = pd.to_numeric(drug_df["Naloxone_Response"], errors="coerce").fillna(0).astype(int)
    
    # Compute Effectiveness
    total_cases = len(drug_df)
    responsive_cases = drug_df["Naloxone_Response"].sum()

    # Debugging Output
    print(f"🔹 Total Cases: {total_cases}, Responsive Cases: {responsive_cases}")

    # Compute Percentage
    response_percentage = round((responsive_cases / total_cases) * 100, 2)

    return response_percentage



# Prediction Function
def predict_complex_overdose(drug_name):
    if drug_name not in le.classes_:
        st.warning(f"⚠️ '{drug_name}' not in trained vocabulary! Results may be inaccurate.")
        return 0.0, 0.0

    drug_encoded = le.transform([drug_name])[0]
    input_features = np.array([[drug_encoded, 1, 1]])  # Assume fentanyl=1, naloxone_response=1
    prediction = round(model.predict_proba(input_features)[0][1] * 100, 2)  # Round to 2 decimal places
    
    # Get naloxone effectiveness percentage
    response_percentage = get_naloxone_effectiveness(drug_name)
    
    return prediction, response_percentage

# Display Prediction
if st.button("🚀 Predict"):
    risk, response_percentage = predict_complex_overdose(drug_name)
    st.write(f"🔴 **Overdose Risk:** {risk:.2f}%")
    st.write(f"💊 **Naloxone Effectiveness:** {response_percentage:.2f}%")
