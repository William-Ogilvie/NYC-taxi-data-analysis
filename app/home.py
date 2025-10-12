"""
home.py
===========

The home page of the JFK Taxis Streamlit app.
"""

import streamlit as st

st.set_page_config(page_title="JFK Taxis Analysis", layout="wide")

# --- Header ---
st.title("NYC and JFK Airport Taxi Data Analysis")
st.markdown("### An interactive tool for exploring taxi data inside NYC and forecasting taxi pick ups at JFK Airport")

st.markdown("---")

# --- Overview ---
st.header("About This Application")
st.write("""
This application provides tools for analyzing NYC yellow taxi trip data with a specific focus of modelling yellow taxi pick ups at JFK Airport. 
The data spans from 2011 to 2025 and includes millions of trip records.

Use the sidebar to navigate between the **EDA** and **Time Series** pages to explore different aspects of the data.
""")

st.markdown("---")

# --- Two column layout for page descriptions ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("EDA - Interactive Choropleth Maps")
    st.write("""
    **Explore patterns in NYC taxi data across the entire city:**
    
    - **Custom choropleth maps** showing taxi pickups and drop-offs across all the NYC taxi zones
    - **Filter by year and month** to see some of the patterns outlined in 1_EDA.ipynb (2011-2025)
    - **Customizable scale** to allow for more control to investiage different parts of NYC 
    - **Borough and zone filtering** to focus on specific areas by dropping high taxi pick up/drop off boroughs/zones
    
    Perfect for understanding the geographic distribution and identifying hotspots 
    in taxi demand across different time periods.
    """)
    
    if st.button("Go to EDA Page"):
        st.switch_page("pages/eda.py")

with col2:
    st.subheader("Time Series - Forecasting & Analysis")
    st.write("""
    **Build and evaluate custom forecasting models for JFK Airport Yellow taxi pick ups:**
    
    - **Multiple model types**: Linear, Non-linear (XGBoost), and Hybrid models (linear regressions boosted on the residuals with XGBoost)
    - **Custom feature selection**: Choose your own lags and Fourier features
    - **Multi step forecasting**: Compare model performance with custom multi step forecasts, including custom start position in the test series and custom forecast length
    - **Model performance**: Computed as MAE for each forecast, there is the option of including a naive baseline to compare the models against
    - **SHAP analysis**: Identify the top 30 most important features by mean absolute SHAP value
    - **Daily or hourly** time series type
    
    Ideal for developing and comparing predictive models to forecast future taxi demand. 
    """)
    
    if st.button("Go to Time Series Page"):
        st.switch_page("pages/time_series.py")

st.markdown("---")

# --- Data Info ---
st.header("Dataset Information")
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Data Source", "NYC TLC")
    st.caption("NYC Taxi & Limousine Commission")

with col_info2:
    st.metric("Date Range", "2011-2025")
    st.caption("14+ years of trip records")

with col_info3:
    st.metric("Focus Area", "JFK Airport")
    st.caption("Yellow taxi trips only")

st.markdown("---")

# --- Quick Start Guide ---
with st.expander("Quick Start Guide"):
    st.markdown("""
    **New to the app? Follow these steps:**
    
    1. **Start with EDA Page**: Get familiar with the geographic distribution of taxi trips
       - Select a year and month to explore
       - Try different scales to find patterns
       - Drop boroughs and zones to focus on specific areas
    
    2. **Move to Time Series Page**: Build forecasting models
       - Add a model by selecting time series type (daily/hourly), model type, and features
       - Train your models using the train buttons
       - Choose your forecast step and offsets (where to start the forecast from inside the test series, offset of 1 starts 1 day into the daily test series)
       - Plot forecasts to see how well your models perform
       - Use SHAP analysis to understand feature importance
    
    3. **Experiment**: Try different model configurations and compare results!
    """)

with st.expander("Notes & Tips"):
    st.markdown("""
    - **Full default features** include all significant lags and fourier features found in 3_EDA_JFK.ipynb
    - **Reduced default features** are based on SHAP analysis from previous models details in 5_model_selection.ipynb
    - **Custom features** let you specify exactly which lags and Fourier features to use
    - **Naive baseline** the naive baseline used is essentially a lag of 1, so it predicts the current value to be the value one time step into the past
    """)

