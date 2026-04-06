# 🌍 Hybrid Atmospheric Intelligence Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link-goes-here.streamlit.app/)

An advanced, real-time Air Quality Index (AQI) prediction and monitoring dashboard. This project validates Machine Learning estimations against real-world hardware sensors to provide accurate environmental and public health intelligence.

## ✨ What We Built

This platform bridges the gap between software prediction and physical measurement:
* **Machine Learning Engine:** Powered by a hyperparameter-tuned XGBoost Regressor to estimate AQI based on temperature, humidity, wind speed, and heuristic traffic/industrial indices.
* **Hybrid Validation:** Directly compares the XGBoost software predictions against physical ground-truth sensors for real-time accuracy checking.
* **Live API Integration:** Fetches real-time meteorological data from OpenWeatherMap and hardware sensor data from the World Air Quality Index (WAQI) global network.
* **Geospatial Mapping:** Features interactive Folium heatmaps displaying localized threat radiuses for target cities.
* **Public Health Analytics:** Translates raw AQI scores into equivalent daily cigarette consumption and provides targeted, actionable advice for athletes, parents, and commuters.
* **Research Pipeline:** Includes comprehensive Jupyter Notebooks documenting Exploratory Data Analysis (EDA), feature engineering, and algorithm comparison (Linear Regression, Random Forest, SVR, ANN, and XGBoost).

## 🛠️ Tech Stack
* **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Engineering:** Pandas, NumPy
* **Geospatial Visualization:** Folium, Streamlit-Folium
* **APIs:** OpenWeatherMap API, WAQI API

## 📁 Project Structure
```text
AQI_Prediction_Project/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules for security
├── models/
│   └── xgboost_aqi_model.pkl   # Trained and optimized XGBoost model
├── data/
│   └── aqi_dataset.csv         # Historical dataset used for model training
└── notebooks/
    ├── 01_data_exploration.ipynb  # EDA and correlation mapping
    └── 02_model_training.ipynb    # Algorithm comparison and Grid Search tuning