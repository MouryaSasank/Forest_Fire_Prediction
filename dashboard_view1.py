# ===============================
# Forest Fire Risk Prediction Dashboard (Polished Version)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="wide"
)

st.markdown(
    """
    <style>
    .metric-box {
        padding: 20px;
        border-radius: 12px;
        color: white;
        font-size: 20px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD & TRAIN MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource

def load_interface_model():
    try:
        df = pd.read_csv("forestfires.csv")

        # NOTE:
        # For deployment and real-time simulation, only core meteorological
        # variables are used, as fire indices and categorical features
        # may not be readily available in live monitoring systems.
        X = df[['temp', 'RH', 'wind', 'rain']]
        y = np.log1p(df['area'])  # Burned area as severity proxy

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception:
        return None

model = load_interface_model()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title(" Forest Fire Risk Prediction System")
st.markdown("Machine Learning–Based Risk Assessment & Visualization")
st.divider()

if model is None:
    st.error("Dataset not found. Ensure forestfires.csv is in the same folder.")
    st.stop()

# -------------------------------------------------
# LAYOUT
# -------------------------------------------------
left, right = st.columns([1, 2])

# -------------------------------------------------
# LEFT PANEL — CONTROLS & PREDICTION
# -------------------------------------------------
with left:
    st.subheader("Simulation Controls")
    st.caption("Adjust weather parameters to simulate fire risk scenarios")

    temp = st.slider(" Temperature (°C)", 0, 50, 30)
    rh = st.slider(" Relative Humidity (%)", 0, 100, 40)
    wind = st.slider(" Wind Speed (km/h)", 0, 100, 20)
    rain = st.slider(" Rainfall (mm)", 0.0, 10.0, 0.0)

    # -------------------------------------------------
    # FUEL MOISTURE (FWI-INSPIRED HEURISTIC)
    # -------------------------------------------------
    fuel_status = "MODERATE"
    fuel_color = "orange"

    if rain > 0.5 or rh > 80:
        fuel_status = "WET / DAMP (Low Ignition Risk)"
        fuel_color = "green"
    elif temp > 30 and rh < 30:
        fuel_status = "EXTREMELY DRY (Very High Risk)"
        fuel_color = "red"
    elif temp > 25 and rh < 50:
        fuel_status = "DRY (High Risk)"

    st.markdown(
        f"<div class='metric-box' style='background-color:{fuel_color}'>"
        f"Fuel Condition: {fuel_status}</div>",
        unsafe_allow_html=True
    )
    st.caption("Derived from meteorological inputs (FWI-inspired heuristic proxy)")

    # -------------------------------------------------
    # AI PREDICTION
    # -------------------------------------------------
    input_data = pd.DataFrame([[temp, rh, wind, rain]], columns=['temp', 'RH', 'wind', 'rain'])
    log_pred = model.predict(input_data)[0]
    base_area = np.expm1(log_pred)

    # -------------------------------------------------
    # HYBRID PHYSICS LAYER
    # -------------------------------------------------
    # NOTE:
    # Severity multipliers are heuristic rules inspired by
    # wildfire behavior principles (wind-driven spread,
    # moisture suppression). These values are not physical
    # constants but decision-support heuristics.

    multiplier = 1.0

    if wind > 40:
        multiplier += 3.0
    elif wind > 20:
        multiplier += 1.0

    if temp > 35:
        multiplier += 1.0

    if rh < 20:
        multiplier += 0.5

    if rain > 0.5:
        multiplier *= 0.1

    if rh > 70:
        multiplier *= 0.5

    final_area = base_area * multiplier

    # -------------------------------------------------
    # RISK CLASSIFICATION
    # -------------------------------------------------
    if final_area < 2:
        risk = "LOW RISK"
        color = "green"
    elif final_area < 10:
        risk = "MODERATE RISK"
        color = "orange"
    else:
        risk = "CRITICAL RISK"
        color = "red"

    st.markdown(
        f"<div class='metric-box' style='background-color:{color}'>"
        f"{risk}<br>{final_area:.2f} ha predicted</div>",
        unsafe_allow_html=True
    )

    st.caption(f"Severity Multiplier Applied: ×{multiplier:.2f}")

# -------------------------------------------------
# RIGHT PANEL — MAP VISUALIZATION
# -------------------------------------------------
with right:
    st.subheader("Geospatial Impact Visualization")
    st.caption("Illustrative impact zone based on predicted severity")

    # Demonstration location (generic forest region)
    center = [16.15, 78.95]
    m = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")

    radius = 200 if final_area < 1 else (final_area * 50) + 200

    folium.Circle(
        location=center,
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=f"Predicted Burn Area: {final_area:.2f} ha"
    ).add_to(m)

    st_folium(m, width=900, height=520)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.caption("This system provides a decision-support simulation for wildfire risk assessment based on meteorological conditions.")
