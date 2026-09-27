# ============================================================
# STREAMLIT APP - DIABETES PREDICTION
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# LOAD TRAINED MODEL AND SCALER
# ------------------------------------------------------------

model = joblib.load("diabetes_logistic_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------

st.title("Diabetes Prediction using Logistic Regression")

st.write(
    "Enter the patient information below to generate "
    "a diabetes prediction."
)

# ------------------------------------------------------------
# USER INPUTS
# ------------------------------------------------------------

Pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

Glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=250,
    value=120
)

BloodPressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

SkinThickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

Insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

BMI = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

Age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

# ------------------------------------------------------------
# CREATE INPUT DATAFRAME
# ------------------------------------------------------------

input_data = pd.DataFrame({
    "Pregnancies": [Pregnancies],
    "Glucose": [Glucose],
    "BloodPressure": [BloodPressure],
    "SkinThickness": [SkinThickness],
    "Insulin": [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
})

# ------------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------------

if st.button("Predict"):

    # Standardize user input
    input_scaled = scaler.transform(input_data)

    # Generate prediction
    prediction = model.predict(input_scaled)[0]

    # Generate probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(
            f"Prediction: Diabetes\n\n"
            f"Probability: {probability:.2%}"
        )
    else:
        st.success(
            f"Prediction: No Diabetes\n\n"
            f"Probability of diabetes: {probability:.2%}"
        )