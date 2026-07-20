import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns (1).pkl")

# Title
st.title(" Heart Disease Prediction")
st.write("Enter patient details and click Predict.")

# User Inputs

age = st.number_input("Age", 1, 100, 40)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ASY", "ATA", "NAP", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    50,
    250,
    120
)

cholesterol = st.number_input(
    "Cholesterol",
    0,
    600,
    200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar (>120 mg/dl)",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["LVH", "Normal", "ST"]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    50,
    250,
    150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["Yes", "No"]
)

oldpeak = st.number_input(
    "Oldpeak",
    0.0,
    10.0,
    1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Down", "Flat", "Up"]
)

# Prediction Button

if st.button("Predict"):

    # Create empty dataframe
    input_data = pd.DataFrame(
        [[0] * len(columns)],
        columns=columns
    )

    # Numerical Features
    input_data["Age"] = age
    input_data["RestingBP"] = resting_bp
    input_data["Cholesterol"] = cholesterol
    input_data["FastingBS"] = fasting_bs
    input_data["MaxHR"] = max_hr
    input_data["Oldpeak"] = oldpeak

    # Sex Encoding
    if sex == "Male":
        input_data["Sex_M"] = 1
    else:
        input_data["Sex_F"] = 1

    # Chest Pain Encoding
    input_data[f"ChestPainType_{chest_pain}"] = 1

    # ECG Encoding
    input_data[f"RestingECG_{resting_ecg}"] = 1

    # Exercise Angina Encoding
    if exercise_angina == "Yes":
        input_data["ExerciseAngina_Y"] = 1
    else:
        input_data["ExerciseAngina_N"] = 1

    # ST Slope Encoding
    input_data[f"ST_Slope_{st_slope}"] = 1

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1] * 100

    # Result
    st.subheader("Result")

    if prediction == 1:
        st.error(
            f"Heart Disease Detected\n\nRisk: {probability:.2f}%"
        )
    else:
        st.success(
            f" No Heart Disease Detected\n\nConfidence: {100-probability:.2f}%"
        )
