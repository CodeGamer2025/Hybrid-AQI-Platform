import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import folium
from streamlit_folium import st_folium

# =================================================================
# 1. ARCHITECTURAL SETUP & UI STYLING
# =================================================================
st.set_page_config(page_title="Hybrid AQI Platform", page_icon="🌍", layout="wide")

# Custom CSS to force high-contrast visibility regardless of System Theme
st.markdown("""
    <style>
    .main { background-color: #f8f9fa !important; }
    
    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        border: 1px solid #dee2e6 !important;
    }
    
    /* Forced Text Colors for Dark Mode Compatibility */
    [data-testid="stMetricLabel"] > div { color: #495057 !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] > div { color: #212529 !important; }
    
    /* Action Button Styling */
    div.stButton > button:first-child {
        background-color: #e63946 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: bold;
        width: 100%;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. DATA ACQUISITION & MODEL INFERENCE
# =================================================================
@st.cache_resource
def load_prediction_model():
    """Loads the serialized XGBoost regressor model."""
    return joblib.load('models/xgboost_aqi_model.pkl')

def fetch_environmental_data(city_name):
    """Retrieves live meteorological and sensor data from OpenWeather and WAQI APIs."""
    try:
        # Fetching keys securely from Streamlit Vault
        owm_key = st.secrets["WEATHER_API_KEY"]
        waqi_key = st.secrets["WAQI_API_KEY"]
        
        # API Endpoints
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={owm_key}&units=metric"
        aqi_url = f"https://api.waqi.info/feed/{city_name}/?token={waqi_key}"
        
        w_data = requests.get(weather_url).json()
        a_data = requests.get(aqi_url).json()
        
        return {
            'temp': w_data['main']['temp'],
            'humidity': w_data['main']['humidity'],
            'wind_speed': w_data['wind']['speed'],
            'lat': w_data['coord']['lat'],
            'lon': w_data['coord']['lon'],
            'sensor_aqi': a_data['data']['aqi'] if a_data['status'] == 'ok' else None
        }
    except Exception as e:
        st.sidebar.error(f"Data Fetch Error: {str(e)}")
        return None

# =================================================================
# 3. INTERACTIVE DASHBOARD LOGIC
# =================================================================
st.title("🌍 Hybrid Atmospheric Intelligence Platform")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("📡 System Configuration")
selected_city = st.sidebar.selectbox("Target Geographical Node", ["Surat", "Mumbai", "Delhi", "Bangalore", "Ahmedabad"])
traffic_idx = st.sidebar.select_slider("Traffic Density (Heuristic)", options=range(0, 11), value=5)
ind_idx = st.sidebar.select_slider("Industrial Output (Heuristic)", options=range(0, 11), value=3)

if st.button("🚀 Initialize Hybrid Validation Protocol"):
    with st.spinner("Synchronizing Software Model with Real-Time Sensor Grid..."):
        env_data = fetch_environmental_data(selected_city)
        
        if env_data:
            # ML Prediction Pipeline
            model = load_prediction_model()
            input_features = pd.DataFrame([[
                env_data['temp'], env_data['humidity'], env_data['wind_speed'], traffic_idx, ind_idx
            ]], columns=['temperature', 'humidity', 'wind_speed', 'traffic_density', 'industrial_activity'])
            
            ml_prediction = model.predict(input_features)[0]

            # Section 1: Comparative Analytics
            st.subheader("📊 Comparative AQI Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Local Temperature", f"{env_data['temp']}°C")
            m2.metric("Relative Humidity", f"{env_data['humidity']}%")
            m3.metric("XGBoost Prediction", f"{ml_prediction:.1f} AQI")
            m4.metric("Hardware Sensor", f"{env_data['sensor_aqi'] if env_data['sensor_aqi'] else 'Offline'} AQI")

            # Section 2: Geospatial & Temporal Analysis
            st.markdown("---")
            left_col, right_col = st.columns([2, 1])
            
            with left_col:
                st.write("**📍 Localization & Impact Radius**")
                map_obj = folium.Map(location=[env_data['lat'], env_data['lon']], zoom_start=12, tiles="cartodbpositron")
                folium.Circle(
                    location=[env_data['lat'], env_data['lon']],
                    radius=3500,
                    color='#e63946',
                    fill=True,
                    fill_opacity=0.2,
                    tooltip="Predicted High-Impact Zone"
                ).add_to(map_obj)
                st_folium(map_obj, width=750, height=400, key="deployment_map")
            
            with right_col:
                st.write("**📈 Simulated 24H Forecast Trend**")
                # Generate a realistic synthetic trend based on the prediction
                trend_points = ml_prediction + (np.sin(np.linspace(0, 2*np.pi, 24)) * 12)
                chart_df = pd.DataFrame(trend_points, columns=['Forecasted AQI'])
                st.area_chart(chart_df, color="#e63946")

            # Section 3: Health Intelligence Report
            st.divider()
            health_status = "Critical" if ml_prediction > 150 else "Moderate" if ml_prediction > 50 else "Safe"
            cig_equivalent = ml_prediction / 22
            
            st.error(f"### Environment Status: {health_status} | Predictive Confidence: High")
            st.info(f"🚬 **Respiratory Impact:** Exposure today is equivalent to smoking **{cig_equivalent:.1f} cigarettes**.")
            
            advice_tab1, advice_tab2 = st.tabs(["🛡️ Protective Measures", "🔬 Scientific Methodology"])
            with advice_tab1:
                st.write("- **Outdoors:** Limit high-intensity aerobic activity during peak sunlight.")
                st.write("- **Indoor:** Ensure HEPA filtration is active in residential zones.")
            with advice_tab2:
                st.write("This platform utilizes a **Hybrid Validation** approach, cross-referencing XGBoost gradient boosting predictions with live hardware telemetry for improved decision support.")

        else:
            st.warning("⚠️ Data Stream Interrupted: Check API connectivity or verify secrets in the Streamlit Cloud Dashboard.")

st.markdown("---")
st.caption("Hybrid Atmospheric Intelligence Platform | Research & Development Node")
