# app/streamlit_app.py
import streamlit as st
from streamlit_folium import st_folium, folium_static
from jfk_taxis import load_geo_data_and_zone_lookup, create_app_choropleths
import numpy as np

st.set_page_config(page_title="JFK Taxis", layout="wide")

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

    set_update_map()

def update_slider():
    scale = st.session_state.scale_slider_widget
    num_cats = st.session_state.num_scale_cats_widget

    st.session_state.params["scale"] = list(np.arange(0, scale + 1, scale // num_cats).tolist())
    set_update_map()

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
with st.sidebar:
    params = st.session_state.params
    st.radio("Pick Up or Drop Off", ("Pick Up", "Drop Off"), on_change = update_pickup_or_dropoff, key = "pickup_choice_widget")
    params["year"] = st.selectbox("Year", [i for i in range(2011, 2026)], on_change=set_update_map)
    params["month"] = st.selectbox("Month", [f"{i:02d}" for i in range(1, 13)], index = 0, on_change=set_update_map)
    st.slider("Max Scale", 0, 700000, 700000, key = "scale_slider_widget", on_change = update_slider)
    st.selectbox("Number scale categories", [i for i in range(2, 9)], index= 5, key = "num_scale_cats_widget", on_change = update_slider)  
    params["drop_boroughs"] = st.multiselect("Boroughs to Drop", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"], default=[], on_change=set_update_map)
    params["drop_ids"] = st.multiselect("Zone IDs to Drop", list(st.session_state.zone_lookup['LocationID'].unique()), default=[], on_change=set_update_map)



# --- Build map ---
if "main_map" not in st.session_state or st.session_state.update_map == True:
    m = build_map()
    st.session_state.update_map = False


# -- Display map ---
folium_static(m, width=1200, height=800)
