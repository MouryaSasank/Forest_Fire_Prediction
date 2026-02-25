# Forest Fire Risk Prediction System

An AI-powered wildfire susceptibility assessment and real-time simulation dashboard built with **Streamlit** and **Random Forest Regression**.

---

## Project Overview

This academic mini-project predicts potential forest fire spread area based on meteorological conditions. It combines a machine learning model with FWI-inspired (Fire Weather Index) heuristic multipliers to produce an interpretable, interactive risk assessment dashboard.

---

## Features

- **Real-time simulation** — Adjust weather sliders and instantly see the fire risk change
- **AI Prediction** — Random Forest Regressor trained on the UCI Forest Fires dataset
- **Fuel Moisture Indicator** — FWI-inspired heuristic to assess forest fuel condition
- **Geospatial Visualization** — Interactive Folium map showing the simulated burn zone
- **Risk Classification** — LOW / MODERATE / CRITICAL risk categories with colour-coded display
- **Dark-themed UI** — Professional dashboard styled for academic presentation

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| Streamlit | Web dashboard framework |
| scikit-learn | Random Forest model |
| Folium + streamlit-folium | Interactive map |
| Pandas / NumPy | Data processing |

---

## Project Structure

```
Mini_Project/
|
|-- streamlit_dashboard.py            # Main dashboard (run this)
|-- Forest_fire_impact_prediction.py  # Model training and analysis scripts
|-- forestfires.csv                   # UCI Forest Fires dataset
|-- requirements.txt                  # Python dependencies
|-- README.md                         # This file
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the dashboard
```bash
streamlit run streamlit_dashboard.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Dataset

**UCI Forest Fires Dataset** — Contains meteorological and spatial data from the Montesinho Natural Park, Portugal.

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/forest+fires)
- **Features used:** Temperature, Relative Humidity, Wind Speed, Rainfall
- **Target:** Burned area (log-transformed)

---

## Methodology

1. **Model:** Random Forest Regressor (100 estimators) trained on log-transformed burned area
2. **Enhancement:** Physics-inspired heuristic multipliers for wind, humidity, and rainfall effects
3. **Risk Thresholds:**
   - LOW RISK — Predicted area < 2 ha
   - MODERATE RISK — 2 to 10 ha
   - CRITICAL RISK — > 10 ha

---

## Disclaimer

This system is designed for **educational and research purposes only**. Predictions are based on historical data and simplified heuristics. This tool should **not** replace official fire danger rating systems or emergency management protocols.

---

## Author

Mini Project | Academic Submission
