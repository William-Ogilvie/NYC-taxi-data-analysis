"""
shap_helpers
=================

This module contains helper functions for working with SHAP values for our models.

"""

# --- Imports ---
from .loading_helpers import load_config
from .training_helpers import load_design
from sklearn.linear_model import LinearRegression
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess
from xgboost import XGBRegressor
import cupy as cp
import numpy as np  

# --- Load config ---
config, PROJECT_ROOT = load_config()


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