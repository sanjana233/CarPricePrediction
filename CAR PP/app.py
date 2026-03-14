"""
=============================================================================
    Car Price Prediction – Streamlit Web Application   (Phase 7)
    ============================================================
    A clean, user-friendly interface for predicting used-car prices.
    All predictions displayed in Indian Rupees (₹).

    Run:  streamlit run app.py
=============================================================================
"""

# ── Currency conversion (USD → INR) ──
USD_TO_INR = 85.0

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── project imports ──
sys.path.insert(0, os.path.dirname(__file__))
from src.utils import clean_data

# ──────────────────────────────────────────────
#  PAGE CONFIG & STYLING
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Car Price Predictor (₹ INR)",
    page_icon="🚗",
    layout="centered",
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        padding: 0.6em 2em;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .price-result {
        font-size: 42px;
        font-weight: 700;
        color: #e8831a;
        text-align: center;
        padding: 20px;
    }
    .header-text {
        text-align: center;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_car_price_model.joblib")


@st.cache_resource
def load_model():
    """Load the pre-trained model (cached so it loads only once)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_model()


# ──────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────
st.markdown("<h1 class='header-text'>🚗 Car Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='header-text' style='color:#777; margin-bottom:30px;'>"
            "Enter the car details below and click <b>Predict</b> to get an estimated price.</p>",
            unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found!  Run `python src/training.py` first to train and save the model.")
    st.stop()

# ── Sidebar / Inputs ──
st.sidebar.header("🔧 Car Specifications")

# -- Manufacturer --
manufacturers = [
    "TOYOTA", "HYUNDAI", "MERCEDES-BENZ", "BMW", "LEXUS", "FORD",
    "HONDA", "CHEVROLET", "NISSAN", "VOLKSWAGEN", "KIA", "SUBARU",
    "AUDI", "SSANGYONG", "OPEL", "MAZDA", "JEEP", "DODGE", "FIAT",
    "MITSUBISHI", "LAND ROVER", "PORSCHE", "ACURA", "DAEWOO",
    "INFINITI", "BUICK", "GMC", "LINCOLN", "SUZUKI", "MINI", "OTHER"
]
manufacturer = st.sidebar.selectbox("Manufacturer", manufacturers, index=0)

# -- Production Year --
prod_year = st.sidebar.slider("Production Year", 1990, 2025, 2015)

# -- Category --
categories = ["Sedan", "Jeep", "Hatchback", "Coupe", "Universal",
              "Minivan", "Microbus", "Pickup", "Cabriolet", "Goods wagon", "Limousine"]
category = st.sidebar.selectbox("Category", categories, index=0)

# -- Fuel type --
fuel_types = ["Petrol", "Diesel", "Hybrid", "LPG", "CNG", "Plug-in Hybrid", "Hydrogen"]
fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types, index=0)

# -- Engine volume --
engine_vol = st.sidebar.slider("Engine Volume (L)", 0.5, 6.5, 2.0, step=0.1)

# -- Turbo --
turbo = st.sidebar.checkbox("Turbo Engine", value=False)

# -- Mileage --
mileage = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=2_000_000, value=100_000, step=5000)

# -- Cylinders --
cylinders = st.sidebar.selectbox("Cylinders", [1, 2, 3, 4, 5, 6, 8, 10, 12, 16], index=3)

# -- Gear box type --
gear_types = ["Automatic", "Tiptronic", "Manual", "Variator"]
gear = st.sidebar.selectbox("Gear Box", gear_types, index=0)

# -- Drive wheels --
drive_options = ["Front", "Rear", "4x4"]
drive = st.sidebar.selectbox("Drive Wheels", drive_options, index=0)

# -- Doors --
doors = st.sidebar.selectbox("Doors", [3, 5, 6], index=1)

# -- Wheel --
wheel_options = ["Left wheel", "Right-hand drive"]
wheel = st.sidebar.selectbox("Wheel Position", wheel_options, index=0)

# -- Color --
colors = ["Black", "White", "Silver", "Grey", "Blue", "Red",
          "Brown", "Green", "Golden", "Orange", "Beige", "Carnelian red",
          "Sky blue", "Yellow", "Purple"]
color = st.sidebar.selectbox("Color", colors, index=0)

# -- Leather interior --
leather = st.sidebar.selectbox("Leather Interior", ["Yes", "No"], index=0)

# -- Airbags --
airbags = st.sidebar.slider("Airbags", 0, 16, 8)

# -- Levy --
levy = st.sidebar.number_input("Levy (tax)", min_value=0, max_value=10_000, value=700, step=50)


# ──────────────────────────────────────────────
#  PREDICTION
# ──────────────────────────────────────────────

def build_input():
    """Assemble user inputs into a DataFrame matching training schema."""
    current_year = 2026
    data = {
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Prod. year": prod_year,
        "Category": category,
        "Leather interior": 1 if leather == "Yes" else 0,
        "Fuel type": fuel_type,
        "Engine volume": engine_vol,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Gear box type": gear,
        "Drive wheels": drive,
        "Doors": doors,
        "Wheel": wheel,
        "Color": color,
        "Airbags": airbags,
        "Turbo": 1 if turbo else 0,
        "Car_Age": current_year - prod_year,
    }
    return pd.DataFrame([data])


# ── Main area ──
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮  Predict Price")

if predict_btn:
    input_df = build_input()
    
    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)  # price can't be negative
        price_inr = prediction * USD_TO_INR
        
        st.markdown("---")
        st.markdown("### 🎯 Estimated Price")
        st.markdown(f"<div class='price-result'>₹ {price_inr:,.0f}</div>", unsafe_allow_html=True)
        st.caption(f"(Approx. ${prediction:,.0f} USD × {USD_TO_INR} = ₹{price_inr:,.0f})")
        
        # Show the input summary in a nice table
        st.markdown("---")
        with st.expander("📋 Input Summary", expanded=False):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ── Footer ──
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:13px;'>"
    "Built with ❤️ using Scikit-Learn & Streamlit  •  Car Price Prediction ML Project"
    "</p>",
    unsafe_allow_html=True
)
