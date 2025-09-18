import streamlit as st
from jfk_taxis import load_config, load_ts_data, split_test_train_sets, create_train_save_models, load_models, run_forecasts_app, compute_shap_values
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt


# --- Load config ---
config, PROJECT_ROOT = load_config()

# --- Constants ---
DAILY_LINEAR_SIG = "daily_linear_models"
DAILY_NON_LINEAR_SIG = "daily_non_linear_models"
HOURLY_LINEAR_SIG = "hourly_linear_models"
HOURLY_NON_LINEAR_SIG = "hourly_non_linear_models"



st.set_page_config(page_title="Time series", layout="wide")



# --- Data processing functions ---
def get_training_test_data():
    """Load and split the time series data into training and testing sets, store in session state.
    """    

    ts_daily, ts_hourly = load_ts_data()

    ts_daily_train, ts_daily_test, ts_hourly_train, ts_hourly_test = split_test_train_sets(ts_daily, ts_hourly)

    st.session_state.ts_daily_train = ts_daily_train
    st.session_state.ts_daily_test = ts_daily_test
    st.session_state.ts_hourly_train = ts_hourly_train
    st.session_state.ts_hourly_test = ts_hourly_test

def add_model():
    """Add a new model to the session state.
    """    

    # Check the lags are inputted correctly if custom
    if not check_lags_input():
        return 0

    # Add linear model to linear dict
    if st.session_state.model_type_widget == "Linear":

        # Get lags and fourier features based on input
        lags, fourier_features = filter_linear_input()

        # Get order
        order = st.session_state.trend_order_widget

        # Add model to session state models
        if st.session_state.ts_type_widget == "Daily":
            st.session_state.daily_linear_models[st.session_state.model_name_widget] = (order, lags, fourier_features, "Linear")
        elif st.session_state.ts_type_widget == "Hourly":
            st.session_state.hourly_linear_models[st.session_state.model_name_widget] = (order, lags, fourier_features, "Linear")

    # Add non-linear model to non-linear dict
    elif st.session_state.model_type_widget == "Non-linear":
        
        # Get lags and fourier features based on input
        lags, fourier_features = filter_non_linear_input() 

        # Get order
        order = st.session_state.trend_order_widget

        # Add model to session state models
        if st.session_state.ts_type_widget == "Daily":
            st.session_state.daily_non_linear_models[st.session_state.model_name_widget] = (order, lags, fourier_features, "Non-linear")
        elif st.session_state.ts_type_widget == "Hourly":
            st.session_state.hourly_non_linear_models[st.session_state.model_name_widget] = (order, lags, fourier_features, "Non-linear")

    # Add hybrid model to hybrid dict
    elif st.session_state.model_type_widget == "Hybrid":

        # Get lags and fourier features based on input
        lags, fourier_features = filter_hybrid_input()

        # Get order
        order = st.session_state.trend_order_widget

        # Add model to session state models
        if st.session_state.ts_type_widget == "Daily":
            st.session_state.daily_linear_models[st.session_state.model_name_widget] = (order, lags, fourier_features, "Hybrid")
        elif st.session_state.ts_type_widget == "Hourly":
            st.session_state.hourly_linear_models[st.session_state.model_name_widget] = (order, lags, fourier_features, "Hybrid")


def check_lags_input() -> bool:
    """Check if the custom lags input is valid.

    Returns:
        bool: True if valid, False otherwise.
    """    
    if st.session_state.lags_input_widget == "Custom":
        for lag in st.session_state.custom_lags_widget.split(","):
            try:
                int(lag)
            except ValueError:
                st.error("Custom lags must be comma-separated integers.")
                return False
    return True

def process_custom_fourier_input() -> list[int]:
    """Process the custom fourier input and convert to the correct format.

    Returns:
        list[int]: list of fourier features in the correct format.
    """    


    fourier_features = []

    # .get ensures if empty we don't get an error

    if st.session_state.ts_type_widget == "Daily":
        for feature in st.session_state.get("custom_fourier_daily_widget", []):
            if feature == "Yearly":
                fourier_features.append("YE")
            elif feature == "Weekly":
                fourier_features.append("W")  

    elif st.session_state.ts_type_widget == "Hourly":
        for feature in st.session_state.get("custom_fourier_hourly_widget", []):
            if feature == "Yearly":
                fourier_features.append("YE")
            elif feature == "Weekly":
                fourier_features.append("W") 
            elif feature == "Daily":
                fourier_features.append("D")

    return fourier_features
    
def process_custom_lags_input() -> list[int]:
    """Process the custom lags input and convert to the correct format.

    Returns:
        list[int]: list of lags in the correct format.
    """    
    lags = [int(x) for x in st.session_state.custom_lags_widget.split(",")]

    lags = list(set(lags))  # Remove duplicates
    lags.sort()  
    return lags

def filter_input_template(daily_reduced_model: str, hourly_reduced_model: str) -> tuple[list[int], list[int]]:
    """Template filter for the input based on the selected options.

    Args:
        daily_reduced_model (str): reduced model name for daily time series.
        hourly_reduced_model (str): reduced model name for hourly time series.

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
            lags = config["shap"][daily_reduced_model]["extracted_lags"] 
        elif st.session_state.lags_input_widget == "Custom":
            lags = process_custom_lags_input()

        if st.session_state.fourier_input_widget == "Full default":
            fourier_features = config["modelling"]["daily_fourier_features"]
        elif st.session_state.fourier_input_widget == "Reduced default":
            fourier_features = config["shap"][daily_reduced_model]["extracted_fourier_features"] 
        elif st.session_state.fourier_input_widget == "Custom":
            fourier_features = process_custom_fourier_input()

    elif st.session_state.ts_type_widget == "Hourly":
        if st.session_state.lags_input_widget == "Full default":
            lags = config["modelling"]["hourly_lags"][:config["modelling"]["hourly_num_lags"]]
            extra_lags = config["modelling"]["hourly_extra_lags"]

            for lag in extra_lags:
                if lag not in lags:
                    lags.append(lag)
        elif st.session_state.lags_input_widget == "Reduced default":
            lags = config["shap"][hourly_reduced_model]["extracted_lags"] 
        elif st.session_state.lags_input_widget == "Custom":
            lags = process_custom_lags_input()

        if st.session_state.fourier_input_widget == "Full default":
            fourier_features = config["modelling"]["hourly_fourier_features"]
        elif st.session_state.fourier_input_widget == "Reduced default":
            fourier_features = config["shap"][hourly_reduced_model]["extracted_fourier_features"] 
        elif st.session_state.fourier_input_widget == "Custom":
            fourier_features = process_custom_fourier_input()

    return lags, fourier_features
    

def filter_linear_input() -> tuple[list[int], list[int]]:
    """Filter the linear input based on the selected options.

    Returns:
        tuple[list[int], list[int]]: filtered lags and fourier features.
    """    

    lags = []
    fourier_features = []

    # Run template filter with linear reduced models
    lags, fourier_features = filter_input_template("daily_linear_order2", "hourly_linear_order2")

    return lags, fourier_features
    

    
def filter_non_linear_input() -> tuple[list[int], list[int]]:
    """Filter the non linear input based on the selected options.

    Returns:
        tuple[list[int], list[int]]: filtered lags and fourier features.
    """    

    lags = []
    fourier_features = []

    # Run template filter with non-linear reduced models
    lags, fourier_features = filter_input_template("daily_base_non_linear", "hourly_base_non_linear")

    return lags, fourier_features

def filter_hybrid_input() -> tuple[list[int], list[int]]:
    """Filter the hybrid input based on the selected options.

    Returns:
        tuple[list[int], list[int]]: filtered lags and fourier features.
    """    

    lags = []
    fourier_features = []

    # Run template filter with hybrid reduced models
    lags, fourier_features = filter_input_template("daily_hybrid_order2", "hourly_hybrid_order2")

    return lags, fourier_features

def remove_daily():
    for model_name in st.session_state.daily_linear_models_to_remove_widget:
        if model_name in st.session_state.daily_linear_models:
            del st.session_state.daily_linear_models[model_name]
            # Also remove from trained models if present
            if model_name in st.session_state.daily_trained_linear_models:
                st.session_state.daily_trained_linear_models.remove(model_name) 
    for model_name in st.session_state.daily_non_linear_models_to_remove_widget:
        if model_name in st.session_state.daily_non_linear_models:
            del st.session_state.daily_non_linear_models[model_name]
            # Also remove from trained models if present
            if model_name in st.session_state.daily_trained_non_linear_models:
                st.session_state.daily_trained_non_linear_models.remove(model_name)
    
def remove_hourly():
    for model_name in st.session_state.hourly_linear_models_to_remove_widget:
        if model_name in st.session_state.hourly_linear_models:
            del st.session_state.hourly_linear_models[model_name]
            # Also remove from trained models if present
            if model_name in st.session_state.hourly_trained_linear_models:
                st.session_state.hourly_trained_linear_models.remove(model_name)
    for model_name in st.session_state.hourly_non_linear_models_to_remove_widget:
        if model_name in st.session_state.hourly_non_linear_models:
            del st.session_state.hourly_non_linear_models[model_name]
            # Also remove from trained models if present
            if model_name in st.session_state.hourly_trained_non_linear_models:
                st.session_state.hourly_trained_non_linear_models.remove(model_name)

# --- Modelling functions ---
def hybrid_model():
    """ The default hybrid model using XGBoost for the non-linear component.

    Returns:
        _type_: the XGBoost regressor instance.
    """    


    return XGBRegressor(
        n_estimators=config["xgboost_default"]["n_estimators"],
        learning_rate=config["xgboost_default"]["learning_rate"],
        max_depth=config["xgboost_default"]["max_depth"],
        subsample=config["xgboost_default"]["subsample"],
        colsample_bytree=config["xgboost_default"]["colsample_bytree"],
        random_state=config["xgboost_setup"]["random_state"],
        eval_metric=config["xgboost_setup"]["eval_metric"],
        tree_method=config["xgboost_setup"]["tree_method"],
        device=config["xgboost_setup"]["device"]
    )

def train_daily_linear_models():
    """Train daily linear models. Models are saved using the signature file format as in the notebooks. We save the model names to session state.
    """

    for model_name, (order, lags, fourier_features, model_type) in st.session_state.daily_linear_models.items():

        # Check the model name isn't already in the training set to avoid retraining
        if model_name in st.session_state.daily_trained_linear_models:
            continue

        # Check if hybrid or linear
        if model_type == "Linear":
            hybrid = None
        elif model_type == "Hybrid":
            hybrid = hybrid_model()

        time_step = "D"

        order_list = [order]

        ts_train = st.session_state.ts_daily_train

        # As we are having to train and save each model separtely we are going to need several model signautres, to avoid confusion these will be standarised by a universal prefix and then the model name 
        sig = DAILY_LINEAR_SIG + f"_{model_name}"
       
        # Because each model potentially has different lags/fourier features we need to train/save them separately even though this function is designed to do several models at a time
        create_train_save_models([model_name], [], hybrid, sig, order_list, lags, fourier_features, time_step, ts_train)

        # Append to daily linear trained models
        st.session_state.daily_trained_linear_models.append(model_name)

def train_daily_non_linear_models():
    """Train daily non linear models. Models are saved using the signature file format as in the notebooks. We save the model names to session state.
    """

    for model_name, (order, lags, fourier_features, model_type) in st.session_state.daily_non_linear_models.items():

        # Check the model name isn't already in the training set to avoid retraining
        if model_name in st.session_state.daily_trained_non_linear_models:
            continue


        hybrid = None

        time_step = "D"

        order_list = [order]

        ts_train = st.session_state.ts_daily_train

        # As we are having to train and save each model separtely we are going to need several model signautres, to avoid confusion these will be standarised by a universal prefix and then the model name
        sig = DAILY_NON_LINEAR_SIG + f"_{model_name}"

        # Because each model potentially has different lags/fourier features we need to train/save them separately even though this function is designed to do several models at a time
        create_train_save_models([], [model_name], hybrid, sig, order_list, lags, fourier_features, time_step, ts_train)

        # Append to daily non linear trained models
        st.session_state.daily_trained_non_linear_models.append(model_name)

def check_daily_offsets_input() -> bool:
    """Check if the custom offsets input is valid.

    Returns:
        bool: True if valid, False otherwise.
    """    
    if st.session_state.daily_offsets_widget == "Custom":
        for offset in st.session_state.daily_custom_offsets_widget.split(","):
            try:
                int(offset)
            except ValueError:
                st.error("Custom offsets must be comma-separated integers.")
                return False
            # Check whether offset is in allowed range 
            if offset > config["modelling"]["daily_offset"][-1]:
                st.error(f"Custom offsets must be less than or equal to {config['modelling']['daily_offset'][-1]}.")
                return False
    return True

def check_hourly_offsets_input() -> bool:
    """Check if the custom offsets input is valid.

    Returns:
        bool: True if valid, False otherwise.
    """    
    if st.session_state.hourly_offsets_widget == "Custom":
        for offset in st.session_state.hourly_custom_offsets_widget.split(","):
            try:
                int(offset)
            except ValueError:
                st.error("Custom offsets must be comma-separated integers.")
                return False
            # Check whether offset is in allowed range
            if offset > config["modelling"]["hourly_offset"][-1]:
                st.error(f"Custom offsets must be less than or equal to {config['modelling']['hourly_offset'][-1]}.")
                return False
    return True


def get_daily_offsets() -> list[int]:
    """Get daily offsets.

    Returns:
        list[int]: daily offsets.
    """    


    if st.session_state.daily_offsets_widget == "default":
        return config["modelling"]["daily_offset"]
    elif st.session_state.daily_offsets_widget == "custom":
        if not check_daily_offsets_input():
            return []

        custom_offsets = st.session_state.daily_custom_offsets_widget
        return [int(x) for x in custom_offsets.split(",")]
    
def get_hourly_offsets() -> list[int]:
    """Get hourly offsets.

    Returns:
        list[int]: hourly offsets.
    """    


    if st.session_state.hourly_offsets_widget == "default":
        return config["modelling"]["hourly_offset"]
    elif st.session_state.hourly_offsets_widget == "custom":
        if not check_hourly_offsets_input():
            return []

        custom_offsets = st.session_state.hourly_custom_offsets_widget
        return [int(x) for x in custom_offsets.split(",")]



def plot_selected_daily_models():
    """Plot selected daily models.
    """  

    full_linear_models = {}
    full_non_linear_models = {}


    ts_train = st.session_state.ts_daily_train
    ts_test = st.session_state.ts_daily_test
    time_step = "D"
    naive = st.session_state.daily_naive_widget
    steps = st.session_state.daily_steps_widget
    offset_list = get_daily_offsets()

    for model_name in st.session_state.daily_linear_models_to_plot_widget:

        sig = DAILY_LINEAR_SIG + f"_{model_name}"


        linear_models_loaded, non_linear_models_loaded = load_models(sig)

        full_linear_models = {**full_linear_models, **linear_models_loaded}
        full_non_linear_models = {**full_non_linear_models, **non_linear_models_loaded}

    for model_name in st.session_state.daily_non_linear_models_to_plot_widget:

        sig = DAILY_NON_LINEAR_SIG + f"_{model_name}"

        linear_models_loaded, non_linear_models_loaded = load_models(sig)

        full_linear_models = {**full_linear_models, **linear_models_loaded}
        full_non_linear_models = {**full_non_linear_models, **non_linear_models_loaded}

    
    # Run forecasts
    forecast_figs, bar_plot_figs, avg_bar_plot_fig = run_forecasts_app(steps, offset_list, full_linear_models, full_non_linear_models, naive, time_step, ts_train, ts_test)

    # Update session state with forecasts
    st.session_state.daily_forecast_fig = forecast_figs
    st.session_state.daily_bar_plot_fig = bar_plot_figs
    st.session_state.daily_avg_bar_plot_fig = avg_bar_plot_fig

    # Store the offsets used so we can display for plot
    st.session_state.daily_offsets_used = offset_list

def display_daily_plots():
    offsets = st.session_state.daily_offset_to_display_widget

    forecast_figs = []
    bar_plot_figs = []

    for offset in offsets:
        forecast_figs.append(st.session_state.daily_forecast_fig[offset])
        bar_plot_figs.append(st.session_state.daily_bar_plot_fig[offset])

    st.session_state.daily_forecast_figs_to_display = forecast_figs
    st.session_state.daily_bar_plot_figs_to_display = bar_plot_figs

def train_hourly_linear_models():
    """Train hourly linear models. Models are saved using the signature file format as in the notebooks. We save the model names to session state.
    """

    for model_name, (order, lags, fourier_features, model_type) in st.session_state.hourly_linear_models.items():

        # Check the model name isn't already in the training set to avoid retraining
        if model_name in st.session_state.hourly_trained_linear_models:
            continue

        # Check if hybrid or linear
        if model_type == "Linear":
            hybrid = None
        elif model_type == "Hybrid":
            hybrid = hybrid_model()

        time_step = "h"

        order_list = [order]

        ts_train = st.session_state.ts_hourly_train

        # As we are having to train and save each model separtely we are going to need several model signautres, to avoid confusion these will be standarised by a universal prefix and then the model name
        sig = HOURLY_LINEAR_SIG + f"_{model_name}"

        # Because each model potentially has different lags/fourier features we need to train/save them separately even though this function is designed to do several models at a time
        create_train_save_models([model_name], [], hybrid, sig, order_list, lags, fourier_features, time_step, ts_train)

        # Append to hourly linear trained models
        st.session_state.hourly_trained_linear_models.append(model_name)

def train_hourly_non_linear_models():
    """Train hourly non linear models. Models are saved using the signature file format as in the notebooks. We save the model names to session state.
    """

    for model_name, (order, lags, fourier_features, model_type) in st.session_state.hourly_non_linear_models.items():

        # Check the model name isn't already in the training set to avoid retraining
        if model_name in st.session_state.hourly_trained_non_linear_models:
            continue

        hybrid = None

        time_step = "h"

        order_list = [order]

        ts_train = st.session_state.ts_hourly_train

        # As we are having to train and save each model separtely we are going to need several model signautres, to avoid confusion these will be standarised by a universal prefix and then the model name
        sig = HOURLY_NON_LINEAR_SIG + f"_{model_name}"

        # Because each model potentially has different lags/fourier features we need to train/save them separately even though this function is designed to do several models at a time
        create_train_save_models([], [model_name], hybrid, sig, order_list, lags, fourier_features, time_step, ts_train)

        # Append to hourly non linear trained models
        st.session_state.hourly_trained_non_linear_models.append(model_name)

def plot_selected_hourly_models():
    """Plot selected hourly models.
    """  

    full_linear_models = {}
    full_non_linear_models = {}


    ts_train = st.session_state.ts_hourly_train
    ts_test = st.session_state.ts_hourly_test
    time_step = "h"
    naive = st.session_state.hourly_naive_widget
    steps = st.session_state.hourly_steps_widget
    offset_list = get_hourly_offsets()

    for model_name in st.session_state.hourly_linear_models_to_plot_widget:

        sig = HOURLY_LINEAR_SIG + f"_{model_name}"


        linear_models_loaded, non_linear_models_loaded = load_models(sig)

        full_linear_models = {**full_linear_models, **linear_models_loaded}
        full_non_linear_models = {**full_non_linear_models, **non_linear_models_loaded}

    for model_name in st.session_state.hourly_non_linear_models_to_plot_widget:

        sig = HOURLY_NON_LINEAR_SIG + f"_{model_name}"

        linear_models_loaded, non_linear_models_loaded = load_models(sig)

        full_linear_models = {**full_linear_models, **linear_models_loaded}
        full_non_linear_models = {**full_non_linear_models, **non_linear_models_loaded}

    
    # Run forecasts
    forecast_figs, bar_plot_figs, avg_bar_plot_fig = run_forecasts_app(steps, offset_list, full_linear_models, full_non_linear_models, naive, time_step, ts_train, ts_test)
 
    # Update session state with forecasts
    st.session_state.hourly_forecast_fig = forecast_figs
    st.session_state.hourly_bar_plot_fig = bar_plot_figs
    st.session_state.hourly_avg_bar_plot_fig = avg_bar_plot_fig

    # Store the offsets used so we can display for plot
    st.session_state.hourly_offsets_used = offset_list

def display_hourly_plots():
    offsets = st.session_state.hourly_offset_to_display_widget

    forecast_figs = []
    bar_plot_figs = []

    for offset in offsets:

        forecast_figs.append(st.session_state.hourly_forecast_fig[offset])
        bar_plot_figs.append(st.session_state.hourly_bar_plot_fig[offset])
    
    st.session_state.hourly_forecast_figs_to_display = forecast_figs
    st.session_state.hourly_bar_plot_figs_to_display = bar_plot_figs


# --- SHAP functions ---
def compute_daily_shap_values():
    """Compute SHAP values for the selected daily model. Save this in session state.
    """    

    model_name = st.session_state.daily_shap_model_widget

    if model_name in st.session_state.daily_linear_models:
        sig = DAILY_LINEAR_SIG + f"_{model_name}"
    elif model_name in st.session_state.daily_non_linear_models:
        sig = DAILY_NON_LINEAR_SIG + f"_{model_name}"
    else:
        st.error("Selected model not found.")
        return

    linear_models_loaded, non_linear_models_loaded = load_models(sig)

    # To use compute_shap_values we need to know if model is linear, hybrid or non-linear  
    if len(non_linear_models_loaded) > 0:
        linear = False
        hybrid = False
    else:
        linear = True

        # Check if hybrid 
        if linear_models_loaded[model_name][2] is not None:
            hybrid = True
        else:
            hybrid = False
    
    
    # We can only compute SHAP values for one model at a time
    shap_values, X = compute_shap_values(sig, sig, model_name, linear, hybrid) 

    # Create SHAP summary plot
    shap_summary = shap.summary_plot(shap_values, X, max_display = 30, show = False)
    plt.title(f"SHAP summary plot for {model_name} (top 30 features)")
    fig = plt.gcf() 
    st.session_state.daily_shap_summary_fig = fig

def compute_hourly_shap_values():
    """Compute SHAP values for the selected hourly model. Save this in session state.
    """    

    model_name = st.session_state.hourly_shap_model_widget

    if model_name in st.session_state.hourly_linear_models:
        sig = HOURLY_LINEAR_SIG + f"_{model_name}"
    elif model_name in st.session_state.hourly_non_linear_models:
        sig = HOURLY_NON_LINEAR_SIG + f"_{model_name}"
    else:
        st.error("Selected model not found.")
        return

    linear_models_loaded, non_linear_models_loaded = load_models(sig)

    # To use compute_shap_values we need to know if model is linear, hybrid or non-linear  
    if len(non_linear_models_loaded) > 0:
        linear = False
        hybrid = False
    else:
        linear = True

        # Check if hybrid 
        if linear_models_loaded[model_name][2] is not None:
            hybrid = True
        else:
            hybrid = False
    
    
    # We can only compute SHAP values for one model at a time
    shap_values, X = compute_shap_values(sig, sig, model_name, linear, hybrid) 

    # Create SHAP summary plot
    shap_summary = shap.summary_plot(shap_values, X, max_display = 30, show = False)
    plt.title(f"SHAP summary plot for {model_name} (top 30 features)") 
    fig = plt.gcf()
    st.session_state.hourly_shap_summary_fig = fig

# --- Initialize session state ---
if "daily_linear_models" not in st.session_state:
    st.session_state.daily_linear_models = {}

if "daily_non_linear_models" not in st.session_state:
    st.session_state.daily_non_linear_models = {}

if "hourly_linear_models" not in st.session_state: 
    st.session_state.hourly_linear_models = {}

if "hourly_non_linear_models" not in st.session_state:
    st.session_state.hourly_non_linear_models = {}

if "daily_trained_linear_models" not in st.session_state:
    st.session_state.daily_trained_linear_models = []

if "daily_trained_non_linear_models" not in st.session_state:
    st.session_state.daily_trained_non_linear_models = []

if "hourly_trained_linear_models" not in st.session_state:
    st.session_state.hourly_trained_linear_models = []

if "hourly_trained_non_linear_models" not in st.session_state:
    st.session_state.hourly_trained_non_linear_models = []

if "daily_offsets_used" not in st.session_state:
    st.session_state.daily_offsets_used = []

if "hourly_offsets_used" not in st.session_state:
    st.session_state.hourly_offsets_used = []


if "ts_daily_train" not in st.session_state:
    get_training_test_data()

# Safe defaults for conditional widgets
st.session_state.setdefault("custom_fourier_daily_widget", [])
st.session_state.setdefault("custom_lags_daily_widget", [])



# Outside form for dynamic UI
col_ts_type, col_fourier, col_lags = st.columns([1,1,1])
with col_ts_type:
    st.radio("Time series type", options = ["Daily", "Hourly"], index = 0, key = "ts_type_widget")
with col_fourier:
    st.radio("Fourier features", options = ["Full default", "Reduced default", "Custom"], index = 0, key = "fourier_input_widget")
with col_lags:
    st.radio("Lags", options = ["Full default", "Reduced default", "Custom"], index = 0, key = "lags_input_widget")

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
            st.multiselect("Fourier features", options = ["Yearly", "Weekly"], key = "custom_fourier_daily_widget")
        elif st.session_state.ts_type_widget == "Hourly":
            st.multiselect("Fourier features", options = ["Yearly", "Weekly", "Daily"], key = "custom_fourier_hourly_widget")
    

    submitted = st.form_submit_button("Add model", key="add_model_button")



# --- Add model if submitted ---
if submitted:
    add_model()

# --- Display models ---
col_daily_models, col_hourly_models = st.columns([1,1])
with col_daily_models:
    st.write("Daily linear/hybrid models:")
    st.write(st.session_state.daily_linear_models)
    st.write("Daily non-linear models:")
    st.write(st.session_state.daily_non_linear_models)
with col_hourly_models:
    st.write("Hourly linear/hybrid models:")
    st.write(st.session_state.hourly_linear_models)
    st.write("Hourly non-linear models:")
    st.write(st.session_state.hourly_non_linear_models)

# --- Remove models ---
col_remove_daily, col_remove_hourly = st.columns([1,1])
with col_remove_daily:
    st.multiselect("Select daily linear/hybrid models to remove", options = list(st.session_state.daily_linear_models.keys()), key = "daily_linear_models_to_remove_widget")
    st.multiselect("Select daily non-linear models to remove", options = list(st.session_state.daily_non_linear_models.keys()), key = "daily_non_linear_models_to_remove_widget")
    st.button("Remove selected daily models", on_click = remove_daily, key = "remove_daily_button") 
with col_remove_hourly:
    st.multiselect("Select hourly linear/hybrid models to remove", options = list(st.session_state.hourly_linear_models.keys()), key = "hourly_linear_models_to_remove_widget")
    st.multiselect("Select hourly non-linear models to remove", options = list(st.session_state.hourly_non_linear_models.keys()), key = "hourly_non_linear_models_to_remove_widget")
    st.button("Remove selected hourly models", on_click = remove_hourly, key = "remove_hourly_button")

# --- Train models ---
col_daily_train, col_hourly_train = st.columns([1,1])
with col_daily_train:
    st.button("Train daily linear/hybrid models", on_click = train_daily_linear_models, key = "daily_linear_train_button")
    st.button("Train daily non-linear models", on_click = train_daily_non_linear_models, key = "daily_non_linear_train_button")
with col_hourly_train:
    st.button("Train hourly linear/hybrid models", on_click = train_hourly_linear_models, key = "hourly_linear_train_button")
    st.button("Train hourly non-linear models", on_click = train_hourly_non_linear_models, key = "hourly_non_linear_train_button")

# --- Plot models ---
col_daily_plotting, col_hourly_plotting = st.columns([1,1])
with col_daily_plotting:
    st.multiselect("Select daily linear/hybrid models to plot", options = st.session_state.daily_trained_linear_models, key = "daily_linear_models_to_plot_widget")
    st.multiselect("Select daily non-linear models to plot", options = st.session_state.daily_trained_non_linear_models, key = "daily_non_linear_models_to_plot_widget")
    st.number_input("Number of steps (days) to forecast", min_value = 1, max_value = 365, value = 30, step = 1, key = "daily_steps_widget")
    st.radio("Offsets to start forecast", options = ["default", "custom"], index = 0, key = "daily_offsets_widget") 

    if st.session_state.daily_offsets_widget == "custom":
        st.text_input(f"Enter custom offsets as comma-separated values, max value {config['modelling']['daily_offset'][-1]}", value = "0,7,14,30", key = "daily_custom_offsets_widget")
    
    st.radio("Naive", options = [True, False], index = 0, key = "daily_naive_widget")

    st.button("Plot selected daily models", on_click=plot_selected_daily_models, key = "daily_plot_button")
with col_hourly_plotting:
    st.multiselect("Select hourly linear/hybrid models to plot", options = st.session_state.hourly_trained_linear_models, key = "hourly_linear_models_to_plot_widget")
    st.multiselect("Select hourly non-linear models to plot", options = st.session_state.hourly_trained_non_linear_models, key = "hourly_non_linear_models_to_plot_widget")
    st.number_input("Number of steps (hours) to forecast", min_value = 1, max_value = 8760, value = 168, step = 1, key = "hourly_steps_widget")
    st.radio("Offsets to start forecast", options = ["default", "custom"], index = 0, key = "hourly_offsets_widget")

    if st.session_state.hourly_offsets_widget == "custom":
        st.text_input(f"Enter custom offsets as comma-separated values, max value {config['modelling']['hourly_offset'][-1]}", value = "0,168,336,504,672", key = "hourly_custom_offsets_widget")
    
   
    st.radio("Naive", options = [True, False], index = 0, key = "hourly_naive_widget")

    st.button("Plot selected hourly models", on_click=plot_selected_hourly_models, key = "hourly_plot_button")

# --- Display plots if available ---
col_daily_figures, col_hourly_figures = st.columns([1,1])
with col_daily_figures:
    st.multiselect("Select offset to display on plot", options=st.session_state.daily_offsets_used, key = "daily_offset_to_display_widget")
    st.button("Display plot for selected offsets", on_click = display_daily_plots, key = "daily_display_button")

    if st.session_state.get("daily_forecast_figs_to_display", None) is not None:
        for fig, bar in zip(st.session_state.daily_forecast_figs_to_display, st.session_state.daily_bar_plot_figs_to_display):
            st.pyplot(fig)
            st.pyplot(bar)
        
        st.pyplot(st.session_state.daily_avg_bar_plot_fig)
with col_hourly_figures: 
    st.multiselect("Select offset to display on plot", options=st.session_state.hourly_offsets_used, key = "hourly_offset_to_display_widget")
    st.button("Display plot for selected offsets", on_click = display_hourly_plots, key = "hourly_display_button") 
    
    if st.session_state.get("hourly_forecast_figs_to_display", None) is not None:
        for fig, bar in zip(st.session_state.hourly_forecast_figs_to_display, st.session_state.hourly_bar_plot_figs_to_display):
            st.pyplot(fig)
            st.pyplot(bar)
        
        st.pyplot(st.session_state.hourly_avg_bar_plot_fig)
    

# --- SHAP values ---
col_daily_shap, col_hourly_shap = st.columns([1,1])
with col_daily_shap:
    st.selectbox("Select daily model for SHAP values", options = st.session_state.daily_trained_linear_models + st.session_state.daily_trained_non_linear_models, key = "daily_shap_model_widget")
    st.button("Compute SHAP values for selected daily model", on_click = compute_daily_shap_values, key = "daily_shap_button")
with col_hourly_shap:
    st.write("Be warned that computing SHAP values for hourly models can take a long time due to the size of the dataset. Potentially in the region of 24 minutes at least on my machine.")
    st.selectbox("Select hourly model for SHAP values", options = st.session_state.hourly_trained_linear_models + st.session_state.hourly_trained_non_linear_models, key = "hourly_shap_model_widget")
    st.button("Compute SHAP values for selected hourly model", on_click = compute_hourly_shap_values, key = "hourly_shap_button")

# --- Display SHAP plots if available ---
col_daily_shap_fig, col_hourly_shap_fig = st.columns([1,1])
with col_daily_shap_fig:
    if st.session_state.get("daily_shap_summary_fig", None) is not None:
        st.pyplot(st.session_state.daily_shap_summary_fig)
with col_hourly_shap_fig: 
    if st.session_state.get("hourly_shap_summary_fig", None) is not None:
        st.pyplot(st.session_state.hourly_shap_summary_fig)