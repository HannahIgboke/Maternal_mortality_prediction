import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('gradient_boosting_model.pkl')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Title of the app
st.title("Maternal Mortality Risk Predictor")

# Add an image under the title
st.image("maternal-deaths.jpeg")

# Navigation for sections
menu = st.sidebar.selectbox("Navigate", ["Home", "Predict Risk"])

# Section 1: Home (Introduction and Instructions)
if menu == "Home":
    st.header("Welcome to the Maternal Mortality Risk Predictor")
    st.markdown("""
    Maternal mortality refers to the death of a woman during pregnancy, childbirth, or within 42 days of delivery. 
    It is a major public health concern globally, and early detection of high-risk cases can save lives.
    
    This app is designed to assist doctors and medical professionals in assessing the risk level of maternal mortality based on key medical parameters.
    
    ### How to Use:
    1. Navigate to the **Predict Risk** section using the sidebar.
    2. Enter the required patient information, including age, blood pressure, blood sugar levels, body temperature, and heart rate.
    3. Click **Predict Risk** to get the model's prediction of whether the patient is at low, medium, or high risk.
    
    **Note**: This app is a decision support tool and should be used alongside clinical expertise.
    """)

# Section 2: Predict Risk
elif menu == "Predict Risk":
    st.header("Predict Maternal Mortality Risk")
    st.markdown("Enter the patient's details below:")

    # Input fields
    age = st.number_input("Age (in years)", step=1)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", step=1)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", step=1)
    blood_sugar = st.number_input("Blood Sugar (mmol/L)", step=0.1)
    body_temp = st.number_input("Body Temperature (°F)", step=0.1)
    heart_rate = st.number_input("Heart Rate (bpm)", step=1)

    # Prediction button
    if st.button("Predict Risk"):
        # Apply the scaler to input features before prediction
        features = np.array([[age, systolic_bp, diastolic_bp, blood_sugar, body_temp, heart_rate]])
        scaled_features = scaler.transform(features)

        # Predict using the scaled features
        prediction = model.predict(scaled_features)[0]
        
        # Map prediction to risk level
        risk_mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
        risk_level = risk_mapping[prediction]
        
        # Display the result
        st.subheader(f"Predicted Risk Level: **{risk_level}**")
