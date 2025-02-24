import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
print("Script Started")

# Load dataset
file_path = "../Data/Processed/cleaned_drug_testing_data.csv"
df = pd.read_csv(file_path)

# Drop Benzodiazepine_Present and Nitazene_Present (data leakage risk)
df = df.drop(columns=["Benzodiazepine_Present", "Nitazene_Present"])

# Apply Label Encoding to "Sold as"
le = LabelEncoder()
df["Sold as"] = le.fit_transform(df["Sold as"])

# Train/Test Split by Unique Drugs
unique_drugs = df["Sold as"].unique()
train_drugs, test_drugs = train_test_split(unique_drugs, test_size=0.2, random_state=42)

df_train = df[df["Sold as"].isin(train_drugs)]
df_test = df[df["Sold as"].isin(test_drugs)]

# Prepare Train/Test Data
X_train = df_train[["Sold as", "Fentanyl_Present", "Naloxone_Response"]]
y_train = df_train["Complex_Overdose_Risk"]
X_test = df_test[["Sold as", "Fentanyl_Present", "Naloxone_Response"]]
y_test = df_test["Complex_Overdose_Risk"]

# Calculate Class Weights (Handling Imbalance)
weight_0 = len(y_train) / (2 * sum(y_train == 0))
weight_1 = len(y_train) / (2 * sum(y_train == 1))
scale_pos_weight = weight_0 / weight_1  # XGBoost parameter for class balancing

# Train XGBoost Model
model = XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Create a folder to save models
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

# Save Model and Label Encoder
joblib.dump(model, os.path.join(model_dir, "xgboost_model.pkl"))
joblib.dump(le, os.path.join(model_dir, "label_encoder.pkl"))
print(f"Model and Label Encoder saved in '{model_dir}' folder.")

# Function to Predict Overdose Risk
def predict_complex_overdose(drug_name, fentanyl, naloxone_response):
    drug_encoded = le.transform([drug_name])[0]
    input_features = np.array([[drug_encoded, fentanyl, naloxone_response]])
    prediction = model.predict_proba(input_features)[0][1]  # Get probability
    return round(prediction * 100, 2)  # Return %

# Example Prediction
example_drug = "Down"
probability = predict_complex_overdose(example_drug, fentanyl=1, naloxone_response=1)
print(f"Predicted Likelihood of Complex Overdose for {example_drug}: {probability}%")
