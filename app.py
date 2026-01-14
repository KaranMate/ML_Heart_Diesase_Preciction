import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="❤️",
    layout="centered"
)

# Load saved model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# Title Section
st.markdown(
    """
    <h1 style='text-align: center; color: #e63946;'>❤️ Heart Stroke Prediction</h1>
    <h4 style='text-align: center;'>Developed by <b>Karan</b></h4>
    <hr>
    """,
    unsafe_allow_html=True
)

st.info("Please enter the following health details to assess heart stroke risk.")

# ------------------- User Input Section -------------------

st.subheader("🧍 Personal Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])

with col2:
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])

st.subheader("🫀 Clinical Measurements")

col3, col4 = st.columns(2)
with col3:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col4:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.subheader("📈 Stress Test Result")
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

st.markdown("<hr>", unsafe_allow_html=True)

# ------------------- Prediction -------------------

if st.button("🔍 Predict Heart Risk", use_container_width=True):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    st.markdown("<hr>", unsafe_allow_html=True)

    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease**\n\nPlease consult a cardiologist immediately.")
    else:
        st.success("✅ **Low Risk of Heart Disease**\n\nMaintain a healthy lifestyle!")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: grey;'>
    This prediction is based on a Machine Learning model and should not replace medical advice.
    </p>
    """,
    unsafe_allow_html=True
)
