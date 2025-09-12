import streamlit as st
from jfk_taxis import load_config


# --- Load config ---
config, PROJECT_ROOT = load_config()



st.set_page_config(page_title="Daily time series", layout="wide")


# --- Functions ---
def add_model():
    """Add a new model to the session state.
    """    

    # Check the lags are inputted correctly if custom
    check_lags_input()

    # Add linear model to linear dict
    if st.session_state.model_type_widget == "Linear":

        lags, fourier_features = filter_linear_input()
        
         


        if st.session_state.lags_input_widget == "Full default":
            lags = [1,2,3,4,5,6,7,14,21,28,60,90,180,365]
        elif st.session_state.lags_input_widget == "Reduced default":
            lags = [1,7,14,30,60,90]
        elif st.session_state.lags_input_widget == "None":
            lags = []
        elif st.session_state.lags_input_widget == "Custom":
            lags = [int(x) for x in st.session_state.custom_lags_widget.split(",")]
        else:
            lags = None

        st.session_state.linear_models[st.session_state.model_name_widget] = (st.session_state.trend_order_widget, lags)

def check_lags_input() -> None:
    """Check if the custom lags input is valid.
    """    
    if st.session_state.lags_input_widget == "Custom":
        for lag in st.session_state.custom_lags_widget.split(","):
            try:
                int(lag)
            except ValueError:
                st.error("Custom lags must be comma-separated integers.")

def process_custom_fourier_input() -> list[int]:
    """Process the custom fourier input and convert to the correct format.

    Returns:
        list[int]: list of fourier features in the correct format.
    """    


    fourier_features = []

    for feature in st.session_state.custom_fourier_daily_widget:
        if feature == "Yearly":
            fourier_features.append("YE")
        elif feature == "Weekly":
            fourier_features.append("W")
        elif feature == "Monthly":
            fourier_features.append("M")
        elif feature == "Quarterly":
            fourier_features.append("Q")
        elif feature == "Daily":
            fourier_features.append("D")

        return fourier_features
    
def process_custom_lags_input() -> list[int]:
    """Process the custom lags input and convert to the correct format.

    Returns:
        list[int]: list of lags in the correct format.
    """    
    lags = [int(x) for x in st.session_state.custom_lags_widget.split(",")]
    return lags


def filter_linear_input() -> tuple[list[int], list[int]]:
    """Filter the linear input based on the selected options.

    Returns:
        tuple[list[int], list[int]]: filtered lags and fourier features.
    """    

    lags = []
    fourier_features = []

    # We have different default lags based on time series type
    if st.session_state.ts_type_widget == "Daily":
        if st.session_state.lags_input_widget == "Full default":
            lags = config["modelling"]["daily_lags"]
        elif st.session_state.lags_input_widget == "Reduced default":
            lags = config["shap"]["daily_linear_order2"]["extracted_lags"]
        elif st.session_state.lags_input_widget == "None":
            lags = []
        elif st.session_state.lags_input_widget == "Custom":
            lags = process_custom_lags_input()

        if st.session_state.fourier_input_widget == "Full default":
            fourier_features = config["modelling"]["daily_fourier_features"]
        elif st.session_state.fourier_input_widget == "Reduced default":
            fourier_features = config["shap"]["daily_linear_order2"]["extracted_fourier_features"]
        elif st.session_state.fourier_input_widget == "None":
            fourier_features = []
        elif st.session_state.fourier_input_widget == "Custom":
            fourier_features = process_custom_fourier_input()

    elif st.session_state.ts_type_widget == "Hourly":
        if st.session_state.lags_input_widget == "Full default":
            lags = config["modelling"]["hourly_lags"]
        elif st.session_state.lags_input_widget == "Reduced default":
            lags = config["shap"]["hourly_linear_order2"]["extracted_lags"]
        elif st.session_state.lags_input_widget == "None":
            lags = []
        elif st.session_state.lags_input_widget == "Custom":
            lags = process_custom_lags_input()

        if st.session_state.fourier_input_widget == "Full default":
            fourier_features = config["modelling"]["hourly_fourier_features"]
        elif st.session_state.fourier_input_widget == "Reduced default":
            fourier_features = config["shap"]["hourly_linear_order2"]["extracted_fourier_features"]
        elif st.session_state.fourier_input_widget == "None":
            fourier_features = []
        elif st.session_state.fourier_input_widget == "Custom":
            fourier_features = process_custom_fourier_input()

        return lags, fourier_features
    

# Outside form for dynamic UI
st.radio("Time series type", options = ["Daily", "Hourly"], index = 0, key = "ts_type_widget")
st.radio("Fourier features", options = ["Full default", "Reduced default", "Custom", "None"], index = 0, key = "fourier_input_widget")
st.radio("Lags", options = ["Full default", "Reduced default", "Custom", "None"], index = 0, key = "lags_input_widget")



with st.form("model_form"):
    st.subheader("Add a daily time series model")
    st.text_input("Model name", key = "model_name_widget")
    st.radio("Model type", options = ["Non-linear", "Linear", "Hybrid"], index = 0, key = "model_type_widget")
    st.number_input("Trend order", min_value = 0, max_value = 5, value = 1, step = 1, key = "trend_order_widget")
    # Handle cutom lags and fourier features
    if st.session_state.lags_input_widget == "Custom":
        st.text_input("Enter custom lags as comma-separated values", value = "1,7,14,30,60,90", key = "custom_lags_widget")

    if st.session_state.fourier_input_widget == "Custom":
        if st.session_state.ts_type_widget == "Daily":
            st.multiselect("Fourier features", options = ["Yearly", "Weekly", "Monthly", "Quarterly"], key = "custom_fourier_daily_widget")
        elif st.session_state.ts_type_widget == "Hourly":
            st.multiselect("Fourier features", options = ["Yearly", "Weekly", "Daily", "Quarterly"], key = "custom_fourier_hourly_widget")

    submitted = st.form_submit_button("Add model") 

# --- Add model if submitted ---
if submitted:
    add_model()