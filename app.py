import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("bhooshakthi_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page settings
st.set_page_config(
    page_title="BHOOSHAKTI NER",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 BHOOSHAKTI NER")
st.subheader("AI-Based Landslide Risk Prediction System")

st.write(
    "Enter the environmental conditions below to estimate the landslide risk level."
)

st.divider()

# Input section
st.header("📊 Environmental Data")

col1, col2 = st.columns(2)

with col1:
    rainfall_24h = st.number_input(
        "Rainfall in last 24 hours (mm)",
        min_value=0.0,
        max_value=500.0,
        value=50.0
    )

    rainfall_7d = st.number_input(
        "Rainfall in last 7 days (mm)",
        min_value=0.0,
        max_value=2000.0,
        value=200.0
    )

    soil_moisture = st.number_input(
        "Soil moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    slope = st.number_input(
        "Slope (degrees)",
        min_value=0.0,
        max_value=90.0,
        value=20.0
    )

with col2:
    elevation = st.number_input(
        "Elevation (m)",
        min_value=0.0,
        max_value=5000.0,
        value=500.0
    )

    historical_landslide = st.selectbox(
        "Previous landslide recorded?",
        ["No", "Yes"]
    )

    ground_movement = st.number_input(
        "Ground movement (mm)",
        min_value=0.0,
        max_value=50.0,
        value=2.0
    )

st.divider()

# Prediction button
if st.button("🔍 Predict Landslide Risk", use_container_width=True):

    previous_landslide = 1 if historical_landslide == "Yes" else 0

    input_data = pd.DataFrame([{
        "rainfall_24h": rainfall_24h,
        "rainfall_7d": rainfall_7d,
        "soil_moisture": soil_moisture,
        "slope": slope,
        "elevation": elevation,
        "historical_landslide": previous_landslide,
        "ground_movement": ground_movement
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("🟢 LOW RISK")
        st.write("The model predicts a relatively low landslide risk.")

    elif prediction == 1:
        st.warning("🟡 MEDIUM RISK")
        st.write("The model predicts a moderate landslide risk.")

    else:
        st.error("🔴 HIGH RISK")
        st.write("The model predicts a higher landslide risk. Further assessment is recommended.")

st.divider()

st.caption("BHOOSHAKTI NER — AI-based environmental risk assessment prototype")