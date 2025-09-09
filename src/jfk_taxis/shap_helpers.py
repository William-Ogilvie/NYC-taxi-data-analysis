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

# --- Load config ---
config, PROJECT_ROOT = load_config()




def load_design_for_shap(hyper_dict: dict, type: str) -> dict:
    """ Function will take a dictionary of hyperparamters and load the design matrix for the models specified in that dictionary.

    Args:
        hyper_dict (dict): dictionary of hyperparameters for models (key is model name, values is dict of hyperparameters)
        type (str): daily or hourly to specify which design matrix to load

    Returns:
        dict: dictionary containing the design matrices for each model
    """

    if type == config["shap"]["daily_key"]: 
        # Load the design matrix for daily model
        linear_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["daily_linear"])
    elif type == config["shap"]["hourly_key"]:
        # Load the design matrix for hourly model
        linear_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["hourly_linear"])
    else:
        raise ValueError("Type must be either 'daily' or 'hourly'")

    # Unpack the design, target and deterministic process matrices
    X = non_linear_design_loaded[config["model_naming"]["default_non_linear"]][0]
    y = non_linear_design_loaded[config["model_naming"]["default_non_linear"]][1]
    dp = non_linear_design_loaded[config["model_naming"]["default_non_linear"]][2]


    # Create a dictionary to hold the design, target, dp and hyperparameters for each model
    result_dict = {}    
    
    for key, value in hyper_dict.items():
        result_dict[key] = (X, y, dp, value)

    return result_dict

def fit_linear_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """ Fit a linear regression model to the data and return the fitted model.

    Args:
        X (pd.DataFrame): the design matrix.
        y (pd.Series): the target vecotor.

    Returns:
        LinearRegression: the fitted linear regression model.
    """

    # Convert to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # Define and fit the model
    model = LinearRegression(fit_intercept=False)
    model.fit(X_np, y_np)

    return model

def fit_non_linear_model(X: pd.DataFrame, y: pd.Series, hyperparams: dict) -> XGBRegressor:
    """Fit a non-linear model to the data and return the fitted model.

    Args:
        X (pd.DataFrame): design matrix
        y (pd.Series): target vector
        hyperparams (dict): hyperparameters for the model

    Returns:
        XGBRegressor: the fitted non-linear model
    """    


    # Convert to numpy arrays
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # If using GPU convert to CuPy arrays
    if config["xgboost_setup"]["device"] == "gpu":
        X_cp = cp.asarray(X_np)
        y_cp = cp.asarray(y_np)

        model = XGBRegressor(**hyperparams)
        model.fit(X_cp, y_cp)
    else:
        # Define and fit the model
        model = XGBRegressor(**hyperparams)
        model.fit(X_np, y_np)

    return model



def fit_models_for_shap(design_dict: dict, model_type: str) -> dict:
    """ Fit the models specified in the design_dict and reuturn a dictionary of the fitted models.

    Args:
        design_dict (dict): dictionary containing the design matrix, target, dp and hyperparameters for each model
        model_type (str): type of model to fit (e.g. "non_linear", "linear" or "hybrid")

    Returns:
        dict: dictionary containing the fitted models for each key in design_dict
    """

    # Dictionary to hold the fitted models
    fitted_models = {}    

    for key, value in design_dict.items():
        
        # Unpack the design, target, dp and hyperparameters
        X, y, dp, hyperparams = value
        
        if model_type == "linear":
            model = fit_linear_model(X, y, dp, hyperparams)
            hybrid = None
        elif model_type == "non_linear":
            model = fit_non_linear_model(X, y, dp, hyperparams)
            hybrid = None
        elif model_type == "hybrid":
            model, hybrid = fit_hybrid_model(X, y, dp, hyperparams)
        else:
            raise ValueError("model_type must be either 'linear', 'non_linear' or 'hybrid'") 

        # Store the fitted model in the dictionary
        fitted_models[key] = (model, dp, hybrid)

    return fitted_models       