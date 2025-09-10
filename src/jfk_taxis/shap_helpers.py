"""
shap_helpers
=================

This module contains helper functions for working with SHAP values for our models.

"""

# --- Imports ---
from .loading_helpers import load_config
from .training_helpers import load_design, load_models
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor
import cupy as cp
import numpy as np  
import shap
import matplotlib.pyplot as plt

# --- Load config ---
config, PROJECT_ROOT = load_config()

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

def return_top_30_SHAP(shap_values: np.ndarray, feature_names: pd.Index) ->  list:
    """Get the top 30 features based on SHAP values.

    Args:
        shap_values (np.ndarray): SHAP values from the model
        feature_names (pd.Index): features names corresponding to the SHAP values

    Returns:
        list: list of tuples containing feature names and their corresponding SHAP values
    """    

    # Compute mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis = 0)

    # Get indices of top 30 features (by mean abs SHAP value) (::-1 for descending order)
    top_indices = np.argsort(mean_abs_shap)[-30:][::-1]

    top_features = [(feature_names[i], mean_abs_shap[i]) for i in top_indices]

    return top_features