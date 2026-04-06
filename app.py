import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import folium
from streamlit_folium import st_folium

# =================================================================
# 1. PAGE SETUP & PROFESSIONAL CSS
# =================================================================
st.set_page_config(page_title="Live AQI Platform", page_icon="🌍", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #fdfdfd; }
    .stMetric { 
        background-color: #ffffff; padding: 25px; border-radius: 15px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #f0f2f6;
    }
    /* ORANGE/RED BUTTON FROM SCREENSHOT */
    div.stButton > button:first-child {
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        height: 3em !important;
        padding: 0 30px !important;
        font-weight: bold !important;
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
        # Weather
        w_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        w_res = requests.get(w_url).json()
        t, h, w = w_res['main']['temp'], w_res['main']['humidity'], w_res['wind']['speed']*3.6
        # Hardware Sensor
        s_url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
        s_res = requests.get(s_url).json()
        real_aqi = s_res['data']['aqi'] if s_res['status'] == 'ok' else None
        pm25 = s_res['data']['iaqi'].get('pm25', {}).get('v', 'N/A')
        pm10 = s_res['data']['iaqi'].get('pm10', {}).get('v', 'N/A')
        return t, h, w, real_aqi, pm25, pm10
    except: return 30, 50, 10, None, "N/A", "N/A"

# =================================================================
# 3. MAIN UI
# =================================================================
st.title("🌍 Hybrid AQI Platform")
st.write("Validating **Machine Learning Predictions** against **Real-Time Hardware Sensors**.")
st.divider()

if 'active' not in st.session_state: st.session_state.active = False

city_input = st.selectbox("Select your city to analyze real-time air quality:", list(CITY_COORDS.keys()))

if st.button("Initialize Hybrid Scan"):
    t, h, w, real, p25, p10 = get_data = get_live_data(city_input)
    model = load_model()
    
    # Heuristics
    hr = datetime.now().hour
    traffic = 8.5 if (8<=hr<=10 or 17<=hr<=20) else 4.0
    industry = 80.0 if city_input in ["Vapi", "Delhi", "Ankleshwar"] else 35.0
    stagnation = industry / (w + 0.1)
    
    # Prediction
    feats = pd.DataFrame([[t, h, w, traffic, industry, stagnation]], 
                         columns=['temperature', 'humidity', 'wind_speed', 'traffic_density', 'industrial_activity', 'stagnation_index'])
    pred = model.predict(feats)[0]
    
    st.session_state.results = {'city': city_input, 'pred': pred, 'real': real, 'p25': p25, 'p10': p10, 'stagnation': stagnation}
    st.session_state.active = True

# =================================================================
# 4. DASHBOARD RENDERING (MATCHING SCREENSHOTS)
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
        else:
            st.metric("Ground Truth AQI", "Offline")
            
    with c3:
        st.warning("### Pollutant Breakdown\nPhysical Particulate Matter.")
        st.write(f"**PM 2.5:** {res['p25']} µg/m³")
        st.write(f"**PM 10:** {res['p10']} µg/m³")

    st.divider()
    
    # SECTION 2: MAP & FORECAST
    st.subheader("📊 Geospatial & Temporal Analysis")
    l_col, r_col = st.columns(2)
    
    with l_col:
        st.markdown("**Live Geospatial Heatmap**")
        m = folium.Map(location=CITY_COORDS[res['city']], zoom_start=11, tiles="CartoDB positron")
        folium.CircleMarker(CITY_COORDS[res['city']], radius=40, color="#FF8C00", fill=True, 
                            tooltip=f"{res['city']} AQI: {res['real'] if res['real'] else res['pred']}").add_to(m)
        st_folium(m, width=600, height=350, key=f"map_{res['city']}")
        
    with r_col:
        st.markdown("**Simulated 24-Hour Forecast**")
        chart_data = pd.DataFrame(res['pred'] + np.sin(np.linspace(0, 2*np.pi, 24))*15, columns=['AQI'])
        st.line_chart(chart_data, color="#FF4B4B")

    st.divider()
    
    # SECTION 3: IMPACT
    st.subheader("🚨 Real-World Impact")
    status = "Unhealthy" if res['pred'] > 150 else "Moderate"
    st.error(f"### Predicted AQI: {res['pred']:.0f} ({status})")
    
    cigs = res['pred'] / 22
    st.info(f"🚬 **Breathing this air today is equivalent to smoking {cigs:.1f} cigarettes.**")
    
    st.subheader("📋 Actionable Advice for Today")
    t1, t2, t3 = st.tabs(["🏃 For Athletes", "👶 For Parents", "🚗 For Commuters"])
    with t1: st.write("**Verdict:** Move it indoors. Heavy breathing increases intake by 10x.")
    with t2: st.write("**Verdict:** Keep children inside during peak industrial hours.")
    with t3: st.write("**Verdict:** Normal commute, but keep windows closed in industrial zones.")

    st.divider()
    st.subheader("💡 Why is this happening?")
    if res['stagnation'] > 5:
        st.markdown(f"The **Stagnation Index** in {res['city']} is currently very high. This means industrial emissions are not being blown away by the wind. Consider using public transit today.")
