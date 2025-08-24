import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# =============================
# Streamlit Page Config
# =============================
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# =============================
# Load Model & Data
# =============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("crop_yield_model.pkl")  # replace with your trained model file
        return model
    except:
        return None

model = load_model()

# Optionally load reference data for charts
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("crop_data.csv")  # historical crop dataset
        return df
    except:
        return None

data = load_data()

# =============================
# UI: Header
# =============================
st.title("🌾 Crop Yield Predictor")
st.markdown("""
This tool predicts **crop yield** based on environmental and agricultural inputs.  
Provide your values below and get an instant estimate.
""")

# =============================
# Sidebar: User Input
# =============================
st.sidebar.header("Input Parameters")

states = ["Andhra Pradesh", "Madhya Pradesh", "Uttar Pradesh", "Rajasthan", "Tamil Nadu", "Punjab"]
crops = ["Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Millets"]

state = st.sidebar.selectbox("State", states)
crop = st.sidebar.selectbox("Crop", crops)
year = st.sidebar.slider("Year", 2000, 2030, 2025)

rainfall = st.sidebar.number_input("Annual Rainfall (mm)", min_value=0.0, max_value=4000.0, value=1000.0, step=10.0)
fertilizer = st.sidebar.number_input("Fertilizer Usage (kg/ha)", min_value=0.0, max_value=1000.0, value=200.0, step=5.0)
pesticide = st.sidebar.number_input("Pesticide Usage (kg/ha)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

# =============================
# Prepare Features
# =============================
def preprocess_input(state, crop, year, rainfall, fertilizer, pesticide):
    # Example encoder — replace with your model’s preprocessing pipeline
    state_map = {s: i for i, s in enumerate(states)}
    crop_map = {c: i for i, c in enumerate(crops)}

    features = np.array([[
        year,
        state_map[state],
        crop_map[crop],
        rainfall,
        fertilizer,
        pesticide,
    ]])

    return features

# =============================
# Prediction
# =============================
if model:
    if st.sidebar.button("Predict Yield"):
        X = preprocess_input(state, crop, year, rainfall, fertilizer, pesticide)
        prediction = model.predict(X)[0]

        st.success(f"🌱 Predicted Yield for **{crop}** in **{state} ({year})**: **{prediction:.2f} tons/ha**")

        # Optional: Show related trend if data is available
        if data is not None:
            st.subheader("📊 Historical Trends")
            filtered = data[(data["Crop"] == crop) & (data["State"] == state)]
            if not filtered.empty:
                fig = px.line(filtered, x="Year", y="Yield",
                              title=f"Historical Yield Trend: {crop} in {state}",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)
else:
    st.error("⚠️ Model not found! Please train and save your model as `crop_yield_model.pkl`")
