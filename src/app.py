# src/app.py

import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained pipeline
model_path = os.path.join("models", "best_model.pkl")
model = joblib.load(model_path)

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("💓 Heart Disease Risk Predictor")

def user_input():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 120, 50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 250, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, 0.1)
        ca = st.number_input("Major Vessels Colored (0-3)", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "ST-T abnormality"])
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        slope = st.selectbox("Slope of ST Segment", ["upsloping", "flat", "downsloping"])
    # Create a DataFrame
    input_df = pd.DataFrame({
        "age": [age],
        "trestbps": [trestbps],
        "chol": [chol],
        "thalch": [thalch],
        "oldpeak": [oldpeak],
        "ca": [ca],
        "sex": [sex],
        "cp": [cp],
        "fbs": [fbs],
        "restecg": [restecg],
        "exang": [exang],
        "slope": [slope],
        "thal": [thal]
    })

    return input_df

input_df = user_input()

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Prediction Probability")
    st.write(f"❤️ Risk of Heart Disease: {prob*100:.2f}%")
    st.success("✅ High risk" if prob > 0.5 else "✅ Low risk")
