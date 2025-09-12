import streamlit as st

st.set_page_config(page_title="JFK Taxis", layout="wide")
st.title("Home")
st.markdown("Use the left sidebar to switch pages.")

with st.container():
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Overview")
        st.write("Put your dashboard cards/plots here.")
    with col2:
        st.subheader("Quick Actions")
        st.button("Do thing")

st.markdown("---")
st.caption("Tip: add more pages in the `pages/` folder.")