import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium

# =================================================================
# 1. PAGE SETUP & PROFESSIONAL CSS (DARK MODE FIX)
# =================================================================
st.set_page_config(page_title="Hybrid AQI Platform", page_icon="🌍", layout="wide")

st.markdown("""
    <style>
    /* Force Light Background and Contrast */
    .main { background-color: #fdfdfd !important; }
    
    /* Style Metrics for perfect visibility in Dark Mode */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 20px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        border: 1px solid #f0f2f6 !important;
    }
    
    /* Force Text Colors to be Visible */
    [data-testid="stMetricLabel"] > div { color: #555555 !important; font-weight: bold !important; }
    [data-testid="stMetricValue"] > div { color: #111111 !important; }

    /* Red Button Styling */
    div.stButton > button:first-child {
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        height: 3em !important;
        padding: 0 30px !important;
        font-weight: bold !important;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# =================================================================
# 2. CORE FUNCTIONS & MODELS
# =================================================================
@st.cache_resource
def load_model():
    return joblib.load('models/xgboost_aqi_model.pkl')

WEATHER_API_KEY = st.secrets["WEATHER_API_KEY"]
WAQI_API_KEY = st.secrets["WAQI_API_KEY"]

CITY_COORDS = {
    "Vapi": [20.3893, 72.9106], "Gandhinagar": [23.2156, 72.6369],
    "Ahmedabad": [23.0225, 72.5714], "Surat": [21.1702, 72.8311],
    "Vadodara": [22.3072, 73.1812], "Delhi": [28.7041, 77.1025]
}

def get_live_data(city):
    try:
        w_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        w_res = requests.get(w_url).json()
        t, h, w = w_res['main']['temp'], w_res['main']['humidity'], w_res['wind']['speed']*3.6
        s_url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
        s_res = requests.get(s_url).json()
        real_aqi = s_res['data']['aqi'] if s_res['status'] == 'ok' else None
        pm25 = s_res['data']['iaqi'].get('pm25', {}).get('v', 'N/A')
        pm10 = s_res['data']['iaqi'].get('pm10', {}).get('v', 'N/A')
        return t, h, w, real_aqi, pm25, pm10
    except: return 30, 50, 10, None, "N/A", "N/A"

# =================================================================
# 3. MAIN UI & LOGIC
# =================================================================
st.title("🌍 Hybrid Atmospheric Intelligence Platform")
st.write("Validating **Machine Learning Predictions** against **Real-Time Hardware Sensors**.")
st.divider()

if 'active' not in st.session_state: st.session_state.active = False

city_input = st.selectbox("Select city to analyze real-time air quality:", list(CITY_COORDS.keys()))

if st.button("Initialize Hybrid Scan"):
    t, h, w, real, p25, p10 = get_live_data(city_input)
    model = load_model()
    
    hr = datetime.now().hour
    traffic = 8.5 if (8<=hr<=10 or 17<=hr<=20) else 4.0
    industry = 80.0 if city_input in ["Vapi", "Delhi"] else 35.0
    stagnation = industry / (w + 0.1)
    
    feats = pd.DataFrame([[t, h, w, traffic, industry, stagnation]], 
                         columns=['temperature', 'humidity', 'wind_speed', 'traffic_density', 'industrial_activity', 'stagnation_index'])
    pred = model.predict(feats)[0]
    
    st.session_state.results = {'city': city_input, 'pred': pred, 'real': real, 'p25': p25, 'p10': p10, 'stagnation': stagnation}
    st.session_state.active = True

# =================================================================
# 4. DASHBOARD RENDERING
# =================================================================
if st.session_state.active:
    res = st.session_state.results
    st.subheader("⚖️ System Validation (Model vs. Reality)")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.info("### Software (XGBoost)\nCalculated via meteorological estimation.")
        st.metric("Predicted AQI", f"{res['pred']:.0f}")
    with c2:
        st.success("### Hardware (WAQI)\nPulled from physical city sensors.")
        if res['real']:
            diff = abs(res['real'] - res['pred'])
            st.metric("Ground Truth AQI", f"{res['real']}", delta=f"{diff:.0f} point variance", delta_color="inverse")
        else: st.metric("Ground Truth AQI", "Offline")
    with c3:
        st.warning("### Pollutant Breakdown\nIndividual Air Quality Index (IAQI).")
        st.write(f"**PM 2.5 Index:** {res['p25']}")
        st.write(f"**PM 10 Index:** {res['p10']}")

    st.divider()
    st.subheader("📊 Geospatial & Temporal Analysis")
    l_col, r_col = st.columns(2)
    
    with l_col:
        st.markdown("**Live Geospatial Heatmap**")
        m = folium.Map(location=CITY_COORDS[res['city']], zoom_start=11, tiles="CartoDB positron")
        folium.CircleMarker(CITY_COORDS[res['city']], radius=40, color="#FF8C00", fill=True).add_to(m)
        st_folium(m, width=600, height=350, key=f"map_{res['city']}")
    with r_col:
        st.markdown("**Simulated 24-Hour Forecast**")
        chart_data = pd.DataFrame(res['pred'] + np.sin(np.linspace(0, 2*np.pi, 24))*15, columns=['AQI'])
        st.line_chart(chart_data, color="#FF4B4B")

    st.divider()
    st.subheader("🚨 Real-World Impact")
    status = "Unhealthy" if res['pred'] > 150 else "Moderate"
    cigs = res['pred'] / 22
    st.error(f"### Predicted AQI: {res['pred']:.0f} ({status})")
    st.info(f"🚬 **Exposure is equivalent to smoking {cigs:.1f} cigarettes today.**")
    
    t1, t2 = st.tabs(["🏃 Activity Advice", "🏠 General Safety"])
    with t1: st.write("Move high-intensity workouts indoors. Heavy breathing increases pollutant intake significantly.")
    with t2: st.write("Keep windows closed during peak industrial hours. Use air purifiers if available in the area.")

st.markdown("---")
st.caption("Hybrid Atmospheric Intelligence Platform | Research Node")
