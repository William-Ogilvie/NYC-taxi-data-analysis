"""
eda.py
===========

The EDA page of the JFK Taxis Streamlit app. Creates an interactive choropleth map.
"""

import streamlit as st
from streamlit_folium import folium_static
from jfk_taxis import load_geo_data_and_zone_lookup, create_app_choropleths, load_config
import numpy as np

# --- Load config ---
config, PROJECT_ROOT = load_config()

st.set_page_config(page_title="EDA", layout="wide")

# --- Functions ---
@st.cache_data
def load_geo_data_zone_wrapper():
    geo_data, zone_lookup = load_geo_data_and_zone_lookup()
    return geo_data, zone_lookup

def build_map():
    params = st.session_state.params

    return create_app_choropleths(
        st.session_state.geo_data, 
        st.session_state.zone_lookup,
        params["extra"],
        params["scale"],
        params["year"],
        params["month"],
        params["pickup_or_drop_off"],
        params["drop_boroughs"],
        params["drop_ids"]
    )

def set_update_map():
    st.session_state.update_map = True

def update_pickup_or_dropoff():
    choice = st.session_state.pickup_choice_widget

    if choice == "Pick Up":
        st.session_state.params["pickup_or_drop_off"] = "PU"
    else:
        st.session_state.params["pickup_or_drop_off"] = "DO" 

   

def update_scale():
    scale = st.session_state.max_scale_input_widget
    num_cats = st.session_state.num_scale_cats_widget

    st.session_state.params["scale"] = list(np.arange(0, scale + 1, scale // num_cats).tolist())
    

# --- state init ---
if "params" not in st.session_state:
    st.session_state.params = {
        "year": 2025,
        "month": "01",
        "scale": np.arange(0, 700001, 100000).tolist(),
        "extra": "",
        "drop_boroughs": [],
        "drop_ids": [],
        "pickup_or_drop_off": "PU"
    }

    # load data
    geo_data, zone_lookup = load_geo_data_zone_wrapper()

    st.session_state.geo_data = geo_data
    st.session_state.zone_lookup = zone_lookup

if "update_map" not in st.session_state:
    st.session_state.update_map = True


# --- Controls ---
with st.form("controls"):
    st.subheader("Map Controls (need to hit 'Update Map' button to reload)")
    params = st.session_state.params
    st.radio("Pick Up or Drop Off", ("Pick Up", "Drop Off"), key = "pickup_choice_widget")
    params["year"] = st.selectbox("Year", [i for i in range(2011, 2026)], key = "year_widget")
    params["month"] = st.selectbox("Month", [f"{i:02d}" for i in range(1, 13)], index = 0, key = "month_widget") 
    st.number_input("Max Scale", min_value = 0, max_value = 700000, value = 700000, step = 10000, key = "max_scale_input_widget")  
    st.selectbox("Number scale categories", [i for i in range(3, 9)], index= 4, key = "num_scale_cats_widget") 
    params["drop_boroughs"] = st.multiselect("Boroughs to Drop", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], default=[], key = "drop_boroughs_widget")
    params["drop_ids"] = st.multiselect("Zone IDs to Drop", list(st.session_state.zone_lookup['LocationID'].unique()), default=[], key = "drop_ids_widget")
    submitted = st.form_submit_button("Update Map")

# -- Update map if form submitted ---
if submitted:
    # We need to check the user hasn't tried to enter anything after config["eda"]["max_month_2025"] as we will get an error
    if (params["year"] == 2025) and (int(params["month"]) > config["eda"]["max_month_2025"]):
        st.error(f"Error: For 2025 you can only select months up to {config['eda']['max_month_2025']} as that is all we have data for. Loaded map with month set to {config['eda']['max_month_2025']}.")

        # Reset month to max allowed
        params["month"] = f"{config['eda']['max_month_2025']:02d}" 
    else:
        update_pickup_or_dropoff()
        update_scale()
        set_update_map()

# --- Build map ---
if "main_map" not in st.session_state or st.session_state.update_map == True:
    m = build_map()
    st.session_state.update_map = False


# -- Display map ---
folium_static(m, width=1200, height=800)