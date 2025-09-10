"""
shap_helpers
=================

This module contains helper functions for working with SHAP values for our models.

"""

# --- Imports ---
from .loading_helpers import load_config, save_config
from .training_helpers import load_design, load_models
import pandas as pd
import numpy as np  
import shap
import matplotlib.pyplot as plt
import re

# --- Load config ---
config, PROJECT_ROOT = load_config()

# --- Functions ---
def compute_shap_values(design_sig: str, model_sig: str, model_prefix: str, linear: bool, hybrid: bool) -> tuple[shap.Explainer, pd.DataFrame]:
    """Compute SHAP values for a given model.

    Args:
        design_sig (str): the design signature.
        model_sig (str): the model signature.
        model_prefix (str): the model prefix.
        linear (bool): whether the model is linear.
        hybrid (bool): whether the model is hybrid.

    Returns:
        tuple[np.ndarray, pd.DataFrame]: the SHAP values and the design matrix.
    """    

    # Load designs and models (note for hybrid the hybrid models are in the linear ones)
    linear_design_loaded, non_linear_design_loaded = load_design(design_sig)
    linear_model_loaded, non_linear_model_loaded = load_models(model_sig)

    # Unpack X and the model
    if linear:
        X = linear_design_loaded[model_prefix][0]
        y = linear_design_loaded[model_prefix][1]
        dp = linear_design_loaded[model_prefix][2]

        # If we are in the hybrid case, then the model is actually the hybrid (so index 2) of the tuple
        if hybrid:
            model = linear_model_loaded[model_prefix][2]
        else:
            model = linear_model_loaded[model_prefix][0]
    else:
        X = non_linear_design_loaded[model_prefix][0]
        y = non_linear_design_loaded[model_prefix][1]
        dp = non_linear_design_loaded[model_prefix][2]

        model = non_linear_model_loaded[model_prefix][0]

    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Return SHAP values and the design matrix
    return shap_values, X

def shap_plots(shap_values: shap.Explainer, X: pd.DataFrame, model_name: str) -> None:
    """Create SHAP plots for the given SHAP values and design matrix.

    Args:
        shap_values (np.ndarray): the SHAP values to plot.
        X (pd.DataFrame): the design matrix corresponding to the SHAP values.
        model_name (str): the name of the model (for titles).
    """    
    # Max features to display
    max_features = 30

    # Summary plot
    shap.summary_plot(shap_values, X, max_display=max_features, show = False)
    plt.title(f"SHAP summary plot: top {max_features} features for {model_name}") 
    plt.show()

    # Shap bar plot by mean absolute value
    shap.plots.bar(shap_values, max_display=max_features, show = False)
    plt.title(f"Mean absoulute SHAP values: top {max_features} features for {model_name}")
    plt.show()

def return_top_X_SHAP(shap_values: np.ndarray, feature_names: pd.Index, x: int) ->  list:
    """Get the top 30 features based on SHAP values.

    Args:
        shap_values (np.ndarray): SHAP values from the model
        feature_names (pd.Index): features names corresponding to the SHAP values
        x (int): number of top features to return

    Returns:
        list: list of tuples containing feature names and their corresponding SHAP values
    """    

    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis = 0)

    # Get indices of top x features (by mean abs SHAP value) (::-1 for descending order)
    top_indices = np.argsort(mean_abs_shap)[-x:][::-1]

    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_indices]

    return top_features

def extract_features_from_top_x(top_x: list) -> tuple[list, list, list]:
    """ Extract lags and fourier features from top x SHAP features.

    Args:
        top_x (list): list of tuples of top x features

    Returns:
        tuple[list, list]: list of lags, list of fourier features and list of trends
    """    

    lags = []
    fourier_features = []
    trends = []

    # Loop through and extract lags and fourier features
    for feature, shap_value in top_x:
        if "lag_" in feature:
            lag = int(feature.split("_")[-1]) # strings are formatted as y_lag_x
            lags.append(lag)
        elif "cos" in feature or "sin" in feature:
            # Search for daily ts
            match = re.search(r'=(.*?)-', feature) # strings in daily are formatted as cos(x,freq=YE-DEC)

            if match:
                result = match.group(1)
                fourier_features.append(result)
            else:
                # Search for hourly ts
                match = re.search(r'=(.*?)\)', feature) # strings in hourly are formatted as cos(x,freq=D)

                if match:
                    result = match.group(1)
                    fourier_features.append(result)
        elif "trend" in feature or "const" in feature:
            trends.append(feature)

    # Check that we extracted the right number of features
    if len(lags) + len(fourier_features) + len(trends)!= len(top_x):
        print(lags)
        print(fourier_features)
        print(trends)
        print(top_x)
        raise ValueError("Mismatch in number of features extracted.")
    
    # Now the fourier features will potentially have duplicates, e.g. multiple "W"s from different components of that fourier feature
    # So we just take the unique ones
    fourier_features = list(set(fourier_features))

    return lags, fourier_features, trends

def top_x_feature_extraction(shap_values: np.ndarray, feature_names: pd.Index, x: int) -> tuple[list, list]:
    """ Extract lags and fourier features from top x SHAP features.

    Args:
        shap_values (np.ndarray): SHAP values from the model
        feature_names (pd.Index): features names corresponding to the SHAP values
        x (int): number of top features to return

    Returns:
        tuple[list, list]: list of lags and list of fourier features
    """    

    # Get top x features
    top_x = return_top_X_SHAP(shap_values, feature_names, x)

    # Extract lags and fourier features
    lags, fourier_features, trends = extract_features_from_top_x(top_x)

    return lags, fourier_features, trends

def extract_top_x_features_dict(shap_values_dict: dict, x: int) -> dict:
    """ Extract top x features for each model in the dictionary.

    Args:
        shap_values_dict (dict): dictionary of SHAP values for each model 
        x (int): number of top features to return

    Returns:
        dict: dictionary of tuples of lags, fourier features and trends for each model
    """    

    features_dict = {}

    for key, value in shap_values_dict.items():
        shap_values = value[0].values
        feature_names = value[1].columns

        lags, fourier_features, trends = top_x_feature_extraction(shap_values, feature_names, x)

        # For the lag bufffer we need to ensure that lags is in ascending order
        lags = sorted(lags)

        features_dict[key] = (lags, fourier_features, trends)

    return features_dict

def save_extracted_features_to_config(features_dict: dict, config: dict) -> dict:
    """Save extracted features to the config.

    Args:
        features_dict (dict): dictionary of extracted features
        config (dict): configuration dictionary to update
    """   

    # TODO save locally to config

    for key, value in features_dict.items():
        lags, fourier_features, trends = value

        # Save to config dict

        # Create a new key
        config["shap"][key] = {}
        config["shap"][key]["extracted_lags"] = lags
        config["shap"][key]["extracted_fourier_features"] = fourier_features
        config["shap"][key]["extracted_trends"] = trends

    # Save changes to the yml file
    save_config(config)

    # Load this new config
    config, PROJECT_ROOT = load_config()

    return config 


