import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Forest Fire Risk Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS FOR ENHANCED VISIBILITY
# ============================================================================
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
    }

    /* Main title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Subtitle styling */
    .subtitle {
        font-size: 1.3rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #3b82f6 0%, transparent 100%);
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
    }

    /* Info box */
    .info-box {
        background-color: #1e3a5f;
        border: 2px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        color: #e2e8f0;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .metric-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    /* Risk status card */
    .risk-card {
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    /* Methodology box */
    .methodology-box {
        background-color: #2d3748;
        border: 2px solid #f59e0b;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-top: 1.5rem;
        color: #fbbf24;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Interpretation guide */
    .interpretation-box {
        background-color: #1e3a5f;
        border: 2px solid #60a5fa;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-top: 1rem;
        color: #e2e8f0;
        font-size: 0.95rem;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #7f1d1d;
        border: 3px solid #ef4444;
        padding: 1.25rem;
        border-radius: 0.75rem;
        margin-top: 2rem;
        color: #fecaca;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Footer info boxes */
    .footer-info {
        background-color: #1e293b;
        padding: 1.25rem;
        border-radius: 0.75rem;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.8;
        border: 1px solid #334155;
    }

    /* Multiplier display */
    .multiplier-box {
        background-color: #374151;
        padding: 1rem;
        border-radius: 0.75rem;
        text-align: center;
        font-size: 1.1rem;
        color: #ffffff;
        font-weight: 600;
        border: 2px solid #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================
@st.cache_resource
def load_interface_model():
    """
    Load and train the Random Forest model for fire area prediction.

    Only core meteorological variables are used (temp, RH, wind, rain)
    as fire indices and categorical features may not be readily available
    in live monitoring systems.

    Returns:
        Trained RandomForestRegressor or None if data unavailable.
    """
    try:
        df = pd.read_csv("forestfires.csv")
        X = df[['temp', 'RH', 'wind', 'rain']]
        y = np.log1p(df['area'])   # Log-transform for normalization
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception:
        return None

model = load_interface_model()

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-title">Forest Fire Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Wildfire Susceptibility Assessment & Real-Time Simulation</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong style="font-size: 1.1rem;">Academic Research Project</strong><br><br>
    This system uses machine learning to simulate forest fire spread based on meteorological conditions.
    The model predicts burned area as a proxy for fire risk, enhanced with FWI-inspired heuristic multipliers.
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================================
# ERROR HANDLING
# ============================================================================
if model is None:
    st.error("**Error:** Dataset 'forestfires.csv' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# ============================================================================
# MAIN LAYOUT
# ============================================================================
col_controls, col_map = st.columns([1, 2], gap="large")

# ----------------------------------------------------------------------------
# LEFT COLUMN: SIMULATION CONTROLS
# ----------------------------------------------------------------------------
with col_controls:
    st.markdown('<div class="section-header">Simulation Controls</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #1e293b; padding: 1.25rem; border-radius: 0.75rem;
                margin-bottom: 1.5rem; color: #cbd5e1; border: 1px solid #334155; font-size: 1rem;">
        <strong>Adjust meteorological parameters to simulate fire spread conditions</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT SLIDERS ────────────────────────────────────────────────────────
    temp = st.slider("Temperature (°C)", min_value=0, max_value=50, value=30,
                     help="Ambient air temperature")
    rh   = st.slider("Relative Humidity (%)", min_value=0, max_value=100, value=40,
                     help="Percentage of moisture in the air")
    wind = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, value=20,
                     help="Wind velocity affecting fire spread rate")
    rain = st.slider("Rainfall (mm)", min_value=0.0, max_value=10.0, value=0.0,
                     step=0.1, help="Precipitation amount")

    # ── FUEL MOISTURE CONDITION (FWI-INSPIRED HEURISTIC) ─────────────────────
    st.markdown("---")

    fuel_moisture_status = "MODERATE"
    fuel_color  = "#f97316"   # orange
    fuel_border = "#fb923c"

    if rain > 0.5 or rh > 80:
        fuel_moisture_status = "WET / DAMP (Hard to Burn)"
        fuel_color  = "#22c55e"   # green
        fuel_border = "#4ade80"
    elif temp > 30 and rh < 30:
        fuel_moisture_status = "EXTREMELY DRY (Explosive)"
        fuel_color  = "#ef4444"   # red
        fuel_border = "#f87171"
    elif temp > 25 and rh < 50:
        fuel_moisture_status = "DRY (High Flammability)"
        fuel_color  = "#f97316"   # orange
        fuel_border = "#fb923c"

    st.markdown(f"""
    <div style="background-color: {fuel_color}; border: 3px solid {fuel_border};
                padding: 1.5rem; border-radius: 1rem; text-align: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        <div style="font-size: 0.9rem; color: #ffffff; margin-bottom: 0.75rem;
                    font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">
            INFERRED FOREST FUEL CONDITION
        </div>
        <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            {fuel_moisture_status}
        </div>
        <div style="font-size: 0.85rem; color: #ffffff; margin-top: 0.75rem; opacity: 0.95;">
            FWI-Inspired Heuristic Proxy
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── PREDICTION COMPUTATION ────────────────────────────────────────────────
    # Step 1: Base AI prediction
    input_data = pd.DataFrame([[temp, rh, wind, rain]], columns=['temp', 'RH', 'wind', 'rain'])
    log_pred  = model.predict(input_data)[0]
    base_area = np.expm1(log_pred)

    # Step 2: Physics-inspired heuristic multipliers
    # NOTE: These severity multipliers are heuristic rules inspired by established
    # wildfire behaviour principles (wind-driven spread, moisture suppression).
    # They are decision-support heuristics, not physical constants.
    multiplier = 1.0

    if wind > 40:        multiplier += 3.0
    elif wind > 20:      multiplier += 1.0
    if temp > 35:        multiplier += 1.0
    if rh < 20:          multiplier += 0.5
    if rain > 0.5:       multiplier *= 0.1
    if rh > 70:          multiplier *= 0.5

    final_area = base_area * multiplier

    # Step 3: Risk classification
    if final_area < 2.0:
        risk_status = "LOW RISK / SAFE"
        risk_color  = "#22c55e"
    elif final_area < 10.0:
        risk_status = "MODERATE RISK"
        risk_color  = "#f97316"
    else:
        risk_status = "CRITICAL RISK"
        risk_color  = "#ef4444"

    # ── PREDICTION DISPLAY ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Fire Spread Area</div>
        <div class="metric-value">{final_area:.2f} ha</div>
        <div style="font-size: 0.95rem; opacity: 0.95; margin-top: 0.75rem; font-weight: 500;">
            Hectares (Burned Area Simulation)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="risk-card" style="background-color: {risk_color};">
        {risk_status}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="multiplier-box">
        <strong>Heuristic Severity Multiplier:</strong> ×{multiplier:.2f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="methodology-box">
        <strong style="font-size: 1.05rem;">Methodology</strong><br><br>
        Predicted area represents fire susceptibility. Physics-inspired multipliers adjust for
        wind acceleration, humidity suppression, and precipitation dampening effects.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# RIGHT COLUMN: GEOSPATIAL VISUALIZATION
# ----------------------------------------------------------------------------
with col_map:
    st.markdown('<div class="section-header">Geospatial Impact Simulation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #1e293b; padding: 1.25rem; border-radius: 0.75rem;
                margin-bottom: 1.5rem; color: #cbd5e1; border: 1px solid #334155;
                font-size: 1rem; line-height: 1.6;">
        <strong>Illustrative visualization of potential fire spread zone.</strong><br>
        Location shown is representative (Nallamala Forest, India) and does not indicate an actual ignition point.
    </div>
    """, unsafe_allow_html=True)

    # Map centered on Nallamala Forest, India
    map_center = [16.15, 78.95]
    m = folium.Map(location=map_center, zoom_start=10, tiles='OpenStreetMap')

    viz_radius = 50 if final_area < 1 else (final_area * 50) + 200

    # Fire spread circle
    folium.Circle(
        location=map_center,
        radius=viz_radius,
        color=risk_color,
        fill=True,
        fill_color=risk_color,
        fill_opacity=0.5,
        opacity=1.0,
        weight=4,
        popup=folium.Popup(
            f"""
            <div style="font-family: sans-serif; font-size: 14px;">
                <strong style="font-size: 16px;">Fire Spread Simulation</strong><br><br>
                <strong>Predicted Area:</strong> {final_area:.2f} ha<br>
                <strong>Risk Level:</strong> {risk_status}
            </div>
            """,
            max_width=300
        )
    ).add_to(m)

    # Center marker
    marker_color = 'red' if final_area >= 10 else ('orange' if final_area >= 2 else 'green')
    folium.Marker(
        location=map_center,
        popup=folium.Popup("<strong>Simulation Centre Point</strong>", max_width=200),
        icon=folium.Icon(color=marker_color, icon='fire', prefix='fa')
    ).add_to(m)

    st_folium(m, width=None, height=550)

    st.markdown("""
    <div class="interpretation-box">
        <strong style="font-size: 1.1rem;">Interpretation Guide</strong><br><br>
        <strong>• Circle Size:</strong> Represents simulated burn area extent<br>
        <strong>• Circle Colour:</strong> Indicates risk severity (Green = Low, Orange = Moderate, Red = Critical)<br>
        <strong>• Location:</strong> Representative forest region, not actual fire location<br>
        <strong>• Marker:</strong> Simulation reference point
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.divider()

col_footer1, col_footer2 = st.columns(2, gap="large")

with col_footer1:
    st.markdown("""
    <div class="footer-info">
        <strong style="font-size: 1.05rem; color: #ffffff;">Model Architecture</strong><br><br>
        <strong>Type:</strong> Random Forest Regressor<br>
        <strong>Estimators:</strong> 100 trees<br>
        <strong>Input Features:</strong> Temperature, Humidity, Wind Speed, Rainfall<br>
        <strong>Output:</strong> Log-transformed burned area (hectares)
    </div>
    """, unsafe_allow_html=True)

with col_footer2:
    st.markdown("""
    <div class="footer-info">
        <strong style="font-size: 1.05rem; color: #ffffff;">Enhancement Layer</strong><br><br>
        <strong>Method:</strong> FWI-inspired heuristic multipliers<br>
        <strong>Risk Thresholds:</strong> Low (&lt;2 ha), Moderate (2–10 ha), Critical (&gt;10 ha)<br>
        <strong>Simulation Type:</strong> Real-time meteorological scenario modelling
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong style="font-size: 1.1rem;">ACADEMIC DISCLAIMER</strong><br><br>
    This system is designed for <strong>educational and research purposes only</strong>.
    Predictions are based on historical data and simplified heuristics.
    Fire Weather Index (FWI) components are approximated through meteorological proxies.<br><br>
    <strong>This tool should not replace official fire danger rating systems or emergency management protocols.</strong>
    Geospatial visualizations are illustrative and do not represent actual fire ignition locations
    or guaranteed spread patterns.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #94a3b8; font-size: 0.95rem; font-weight: 500;">
    Forest Fire Risk Prediction System | Academic Machine Learning Project<br>
    <strong>Powered by Random Forest Regressor &amp; Streamlit</strong>
</div>
""", unsafe_allow_html=True)