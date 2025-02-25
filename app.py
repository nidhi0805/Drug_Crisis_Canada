import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 🚀 Load Dataset
file_path = "Data/Processed/drug_testing_data_with_nitazene_presence.csv"
df = pd.read_csv(file_path)
df.fillna("", inplace=True)

# 🚀 Define Target Variable (Overdose Risk)
df["Complex_Overdose_Risk"] = df["Notes"].apply(
    lambda x: 1 if "complex overdose" in x.lower() or "not fully reversible" in x.lower() else 0
)

# 🚀 Drop Potential Leakage Feature
df.drop(columns=["Nitazene_Present"], inplace=True, errors="ignore")

# 🚀 Create `Fentanyl_Present` Feature from `Notes`
df["Fentanyl_Present"] = df["Notes"].apply(lambda x: 1 if "fentanyl" in x.lower() else 0)

# 🚀 Combine Text Columns
combined_text = df["Description"] + " " + df["Notes"]

# 🚀 TF-IDF Feature Engineering
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1200, ngram_range=(1, 3))
text_features = tfidf_vectorizer.fit_transform(combined_text)

# --- Scale TF-IDF features ---
scale_factor = 10
text_features = text_features * scale_factor

# 🚀 Encode Categorical Feature
le_category = LabelEncoder()
df["Category_encoded"] = le_category.fit_transform(df["Category"])

# 🚀 Prepare Features
X_structured = df[["Category_encoded"]].values
X = np.hstack((X_structured, text_features.toarray()))
y = df["Complex_Overdose_Risk"]

# 🚀 Handle Class Imbalance
smote = SMOTE(sampling_strategy=0.75, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 🚀 Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🚀 Train Logistic Regression (Only Logistic Regression is used now)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 🚀 Prediction Function (Uses only Logistic Regression)
def predict_overdose_risk(description, category, notes=""):
    combined_input = description + " " + notes
    text_features_input = tfidf_vectorizer.transform([combined_input]) * scale_factor
    
    # Encode category
    if category in le_category.classes_:
        category_encoded = le_category.transform([category])[0]
    else:
        return "Unknown category. Please enter a valid drug category."

    structured_features = np.array([[category_encoded]])
    input_features = np.hstack((structured_features, text_features_input.toarray()))

    # Always using Logistic Regression
    risk_probability = log_reg.predict_proba(input_features)[0][1]

    return round(risk_probability * 100, 2)

# 🚀 Streamlit UI
st.title("🚑 Overdose Risk Prediction App")
st.write("This app predicts the overdose risk based on drug description, category, and additional notes.")

# 🚀 User Inputs
description = st.text_input("Enter Drug Description", "I found a blue tablet")
category = st.selectbox("Select Drug Category", le_category.classes_)
notes = st.text_area("Additional Notes (Optional)", "This is a small round pill.")

# 🚀 Predict Button (No Model Choice, Always Logistic Regression)
if st.button("Predict Overdose Risk"):
    prediction = predict_overdose_risk(description, category, notes)
    st.success(f" Complex Overdose Risk: {prediction}%")
