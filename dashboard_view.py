import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

#PAGE CONFIGURATION
st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

#1. TRAIN INTERFACE MODEL
@st.cache_resource
def load_interface_model():
    try:
        # Load data
        df = pd.read_csv("forestfires.csv")
        
        # Select features
        X = df[['temp', 'RH', 'wind', 'rain']]
        # NOTE:
        # For deployment and real-time simulation, only core meteorological
        # variables are used, as fire indices and categorical features
        # may not be readily available in live monitoring systems.

        # Use log transformation to normalize area
        y = np.log1p(df['area'])
        
        # Train Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        return None

model = load_interface_model()

# HEADER 
st.title("Forest Fire Risk Prediction")
st.markdown("### Dynamic Wildfire Simulator & Risk Assessment")
st.divider()

if model is None:
    st.error("Error: 'forestfires.csv' not found. Please ensure it is in the same folder.")
    st.stop()


# DASHBOARD LOGIC 
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Simulation Controls")
    st.info("Adjust sliders to see the impact on fire spread.")
    
    # SLIDERS
    temp = st.slider("Temperature (°C)", 0, 50, 30)
    rh = st.slider("Humidity (%)", 0, 100, 40)
    wind = st.slider("Wind Speed (km/h)", 0, 100, 20)
    rain = st.slider("Rainfall (mm)", 0.0, 10.0, 0.0)
    
    # NEW: FUEL MOISTURE INDICATOR (The Answer to your Madam) 
    # We calculate a "Proxy" FFMC (Fine Fuel Moisture Code) to show we know Fuel matters.
    # Logic: Hot + Dry = Low Moisture (Dangerous). Rain = High Moisture (Safe).
    
    fuel_moisture_status = "Moderate"
    fuel_color = "orange"
    
    if rain > 0.5 or rh > 80:
        fuel_moisture_status = "WET / DAMP (Hard to Burn)"
        fuel_color = "green"
    elif temp > 30 and rh < 30:
        fuel_moisture_status = "EXTREMELY DRY (Explosive)"
        fuel_color = "red"
    elif temp > 25 and rh < 50:
        fuel_moisture_status = "DRY (High Flammability)"
        fuel_color = "orange"
        
    st.write("---")
    st.markdown(f"**Inferred Forest Fuel Condition:** :{fuel_color}[**{fuel_moisture_status}**]")
    st.caption("Derived from meteorological inputs (FWI-inspired heuristic proxy)")
    st.write("---")

    # 1. BASE AI PREDICTION
    input_data = pd.DataFrame([[temp, rh, wind, rain]], columns=['temp', 'RH', 'wind', 'rain'])
    log_pred = model.predict(input_data)[0]
    base_area = np.expm1(log_pred) 
    
    #2. HYBRID PHYSICS LAYER
    multiplier = 1.0
    # NOTE:
    # The following severity multipliers are heuristic rules inspired by
    # established wildfire behavior principles (e.g., wind-driven spread,
    # moisture suppression). These values are not physical constants,
    # but decision-support heuristics to enhance interpretability.

    
    # BOOSTERS (For High Risk)
    if wind > 40:
        multiplier += 3.0
    elif wind > 20:
        multiplier += 1.0
        
    if temp > 35:
        multiplier += 1.0
        
    if rh < 20:
        multiplier += 0.5

    # SUPPRESSORS (For Low Risk)
    if rain > 0.5:
        multiplier *= 0.1
    
    if rh > 70:
        multiplier *= 0.5
        
    final_area = base_area * multiplier
    
    # 3. RISK CLASSIFICATION
    if final_area < 2.0:
        status = "LOW / SAFE"
        color = "green"
    elif final_area < 10.0:
        status = "MODERATE RISK"
        color = "orange"
    else:
        status = "CRITICAL RISK"
        color = "red"

    st.metric(label="Predicted Fire Spread", value=f"{final_area:.2f} ha")
    
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {color}; color: white; text-align: center;">
        <h2>{status}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption(f"Physics Severity Multiplier: x{multiplier:.1f}")


with col2:
    st.subheader("Geospatial Impact Simulation")
    
    # MAP CENTERED ON NALLAMALA FOREST (INDIA)
    map_center = [16.15, 78.95] 
    m = folium.Map(location=map_center, zoom_start=10)
    
    # Dynamic Circle Radius
    # If safe, tiny dot. If critical, huge circle.
    viz_radius = 50 if final_area < 1 else (final_area * 50) + 200
    
    folium.Circle(
        location=map_center,
        radius=viz_radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=f"Burn Area: {final_area:.2f} ha"
    ).add_to(m)
    
    st_folium(m, width=800, height=500)