"""
streamlit_app.py - Streamlit UI that calls the FastAPI /predict route.
It does NOT load model.pkl; it talks to the API over HTTP only.

Run (in a second terminal, after starting FastAPI):
    pip install streamlit requests
    streamlit run streamlit_app.py
"""

import requests
import streamlit as st

# 1. Page setup
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter the patient's details and click **Predict**.")

API_URL = "http://127.0.0.1:8000"

# 2. Eight input widgets in columns
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 17, 2)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)

with col2:
    glucose = st.slider("Glucose", 0, 200, 120)
    insulin = st.slider("Insulin", 0, 850, 79)
    age = st.slider("Age", 18, 90, 30)

with col3:
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1)

# 3. Predict button
if st.button("Predict", type="primary"):
    # Exact field names from PatientFeatures in main.py
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "SkinThickness": float(skin_thickness),
        "Insulin": float(insulin),
        "BMI": float(bmi),
        "DiabetesPedigreeFunction": float(dpf),
        "Age": int(age),
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            # 4. Result display
            if result["result_label"] == "Diabetic":
                st.error("Result: Diabetic")
            else:
                st.success("Result: Not Diabetic")

            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        else:
            # 5. Graceful failure (non-200)
            st.warning(
                "The API returned an error. Please make sure FastAPI is running "
                "(`uvicorn main:app --reload`) and the model is loaded, then try again."
            )
    except requests.exceptions.RequestException:
        # 5. Graceful failure (connection problem)
        st.warning(
            "Could not reach the API. Please start FastAPI first with "
            "`uvicorn main:app --reload` and try again."
        )
