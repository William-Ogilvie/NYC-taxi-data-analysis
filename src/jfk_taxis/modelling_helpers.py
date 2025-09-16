"""
modelling_helpers
=================

This module contains helper functions for both creating design matricies, targets, deterministic processes as well as fitting both linear and non linear models to them.
The key function that will be run in the notebook is create_train_save_models which will create the designs, fit the models and save them to pkl files with a given signature
for a list of models names (both linear and non linear). See the function docstring for more details.
"""

# --- Imports ---
from .forecast_helpers import fit_non_linear, preprocess, fit_linear 
from .training_helpers import save_design, save_models
import copy
import pandas as pd
from xgboost import XGBRegressor

# --- Functions ---
def create_design_non_linear(lags: list[int], fourier_features: list[str], time_step: str, ts: pd.Series, name: str) -> dict:
    """ Create design, target and deterministic process for non linear model and return as dict

    Args:
        lags (list[int]): list of lags
        fourier_features (list[str]): list of fourier features
        time_step (str): time step of series so "D" or "h"
        ts (pd.Series): time series itself
        name (str): name of the model for the dictionary

    Returns:
        dict: dictionary containing design, target and deterministic process
    """
    
    # Create non linear design and traget matricies
    (X_non_linear, y_non_linear, dp_non_linear, lags) = preprocess(lags, False, 0, fourier_features, time_step, ts)

    # For the preprocess function the parameters are: lags, constant, order, fourier features, time_step (for the target series), time series

    # Store non linear design matricies
    non_linear_design = {
        name: (X_non_linear, y_non_linear, dp_non_linear, lags)
    }

    return non_linear_design

def create_design_linear(lags: list[int], order: int, fourier_features: list[str], time_step: str, ts: pd.Series, name: str) -> dict:
    """ Create design, target and deterministic process for linear model and return as dict

    Args:
        lags (list[int]): list of lags
        order (int): order of the linear trend
        fourier_features (list[str]): list of fourier features
        time_step (str): time step of series so "D" or "h"
        ts (pd.Series): time series itself
        name (str): name of the model for the dictionary

    Returns:
        dict: dictionary containing design, target and deterministic process
    """

    # Create X,y, dp_linear
    (X,y, dp_linear, lags) = preprocess(lags, True, order, fourier_features, time_step, ts)
    
    # Save as dict 
    linear_design = {
        name: (X, y, dp_linear, lags)
    }

    return linear_design 

def train_non_linear_models(non_linear_design: dict) -> dict:
    """ Trains non linear models on the designs in the dict and returns them

    Args:
        non_linear_design (dict): dictionary containing design, target and deterministic process for each non linear model

    Returns:
        dict: dictionary containing non linear model, deterministic process and None (for hybrid model)
    """

    # Dict for storing non_linear_models
    non_linear_models = {}

    # Loop through design dict and fit non_linear_models
    for key, value in non_linear_design.items():
        non_linear_models[key] = (fit_non_linear(value[0], value[1]), value[2], None, value[3])

    return non_linear_models 

def train_linear_models(linear_design: dict) -> dict:
    """ Trains linear models on the designs in the dict and returns them

    Args:
        linear_design (dict): dictionary containing design, target and deterministic process for each linear model

    Returns:
        dict: dictionary containing linear model, deterministic process and None (for hybrid model)
    """ 

    # Dict for storing linear_models
    linear_models = {}

    # Loop through design dict and fit linear_models
    for key, value in linear_design.items():
        linear_models[key] = (fit_linear(value[0], value[1]), value[2], None, value[3])

    return linear_models

def train_hybrid_models(linear_design: dict, hybrid_model: XGBRegressor) -> dict:
    """ Trains hybrid models on the designs in the dict and returns them

    Args:
        linear_design (dict): dictionary containing design, target and deterministic process for each linear model
        hybrid_model (XGBRegressor): XGBoost regressor model to be used for the hybrid component

    Returns:
        dict: dictionary containing linear model, deterministic process and hybrid model
    """

     
    # Dict for storing hybrid_models
    hybrid_models = {}

    # Loop through design dict and fit hybrid_models
    for key, value in linear_design.items():
        # Unpack X, y and dp 
        X = value[0]
        y = value[1]
        dp = value[2]
        lags = value[3]

        # First fit the linear model
        linear_model = fit_linear(X, y)

        # Get fitted values (convert X to numpy array for prediction)
        X_pred = X.to_numpy()
        y_fit = linear_model.predict(X_pred)

        # Compute resiudals
        y_resid = y - y_fit
 
        # Fit the non linear component to the residuals
        # We need to make a deepcopy of the hybrid model as otherwise we will be just fitting to the same model several times
        hybrid_model_copy = copy.deepcopy(hybrid_model)
        hybrid_model_copy.fit(X, y_resid)

        # Update hybrid models dict, note how we pass the model in two components the linear part and the hybrid part
        # See src/jfk_taxis/forecast_helpers.py to see why
        hybrid_models[key] = (linear_model, dp, hybrid_model_copy, lags)

    return hybrid_models

def create_train_non_linear(names: list[str], lags: list[int], fourier_features: list[str], time_step: str, ts: pd.Series) -> tuple[dict, dict]:
    """ Create design, target, deterministic process, the model and fits the model. Returning two dicts one of designs one of models

    Args:
        names (list[str]): list of names of the non linear models
        lags (list[int]): list of lags
        fourier_features (list[str]): list of fourier features
        time_step (str): time step of series so "D" or "h"
        ts (pd.Series): time series itself

    Returns:
        tuple[dict, dict]: tuple containing two dicts one of designs one of models
    """     

    # Dict of non_linear design, target, dp
    non_linear_design = {}

    # Dict of non_linear models themselves
    non_linear_models = {}

    # Loop through names creating design, target and dp. Fit the models as well
    for name in names:
        # Create design 
        design = create_design_non_linear(lags, fourier_features, time_step, ts, name)

        # Train model
        model = train_non_linear_models(design)

        # Store design and model 
        non_linear_design[name] = design[name]
        non_linear_models[name] = model[name]
    
    return non_linear_design, non_linear_models

def create_train_linear(names: list[str], order_list: list[int], lags: list[int], fourier_features: list[str], time_step: str, ts: pd.Series) -> tuple[dict, dict]:
    """ Create design, target, deterministic process, the model and fits the model. Returning two dicts one of designs one of models

    Args:
        names (list[str]): list of names of the linear models
        order_list (list[int]): list of orders to fit
        lags (list[int]): list of lags
        fourier_features (list[str]): list of fourier features
        time_step (str): time step of series so "D" or "h"
        ts (pd.Series): time series itself

    Returns:
        tuple[dict, dict]: tuple containing two dicts one of designs one of models
    """      

    # Dict of linear design, target, dp
    linear_design = {}

    # Dict of linear models themselves
    linear_models = {}

    # Loop through names creating design, target and dp. Fit the models as well
    for name, order in zip(names, order_list):
        # Create design 
        design = create_design_linear(lags, order, fourier_features, time_step, ts, name)

        # Train model
        model = train_linear_models(design)
        

        # Store design and model 
        linear_design[name] = design[name]
        linear_models[name] = model[name]
    
    return linear_design, linear_models

def create_train_hybrid(names: list[str], hybrid: XGBRegressor, order_list: list[int], lags: list[int], fourier_features: list[str], time_step: str, ts: pd.Series) -> tuple[dict, dict]:
    """ Create design, target, deterministic process, the model and fits the model. Returning two dicts one of designs one of models

    Args:
        names (list[str]): list of names of the hybrid models
        hybrid (XGBRegressor): the non linear part of the model (usually xgboost)
        order_list (list[int]): list of orders to fit
        lags (list[int]): list of lags
        fourier_features (list[str]): list of fourier features
        time_step (str): time step of series so "D" or "h"
        ts (pd.Series): time series itself

    Returns:
        tuple[dict, dict]: tuple containing two dicts one of designs one of models
    """      

    # Dict of hybrid design, target, dp
    hybrid_design = {}

    # Dict of hybrid models themselves
    hybrid_models = {}

    # Loop through names creating design, target and dp. Fit the models as well
    for name, order in zip(names, order_list):
        # Create design 
        design = create_design_linear(lags, order, fourier_features, time_step, ts, name)

        # Train model
        model = train_hybrid_models(design, hybrid)

        # Store design and model 
        hybrid_design[name] = design[name]
        hybrid_models[name] = model[name]
    
    return hybrid_design, hybrid_models

def create_train_save_models(names_linear: list[str], names_non_linear: list[str], hybrid: XGBRegressor | None, sig: str, order_list: list[int], lags: list[int], fourier_features: list[str], time_step: str, ts: pd.Series) -> None:
    """ Creates the designs, trains the models and saves them to pkl files with the given signature

    Args:
        names_linear (list[str]): list of names of the linear models
        names_non_linear (list[str]): list of names of the non linear models
        hybrid (XGBRegressor): the non linear part of the model (usually xgboost)
        sig (str): signature to name the pkl objects when saved (e.g. 5_order_linear_daily)
        order_list (list[int]): list of orders to fit
        lags (list[int]): list of lags
        fourier_features (list[str]): list of fourier features
        time_step (str): time step of series so "D" or "h"
        ts (pd.Series): time series itself
    """      

    # Create copy of ts to avoid overwriting original
    ts_copy = ts.copy()

    # Dict of linear or hybrid designs
    linear_design = {}

    # Dict of linear or hybrid models
    linear_models = {}

    # Dict of non linear designs
    non_linear_design = {}

    # Dict of non_linear models
    non_linear_models = {}

    # First do the case of no hybrid models
    if hybrid is None:
        linear_design, linear_models = create_train_linear(names_linear, order_list, lags, fourier_features, time_step, ts_copy)
    
    else:
        # Even though we are in the hybrid case we will still store them in linear_design and linear_models
        # this is because all "linear models" are actually just hybrid models with None for the hybrid part
        linear_design, linear_models = create_train_hybrid(names_linear, hybrid, order_list, lags, fourier_features, time_step, ts_copy)

    # Create and train non linear models
    non_linear_design, non_linear_models = create_train_non_linear(names_non_linear, lags, fourier_features, time_step, ts_copy)

    # Save designs and models
    save_design(linear_design, non_linear_design, sig)
    save_models(linear_models, non_linear_models, sig)
