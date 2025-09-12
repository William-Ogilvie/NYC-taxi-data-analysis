import streamlit as st

st.set_page_config(page_title="Daily time series", layout="wide")



# --- Add model ---
with st.form("model_form"):
    st.subheader("Add a daily time series model")
