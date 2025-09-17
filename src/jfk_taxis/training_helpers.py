"""
training_helpers
=================

This module contains helper functions for saving models, design matrices, targets, lags and hyperparameters and misc python objects.
The files are saves as pkl files using joblib into the data/saved_objects directory as specified in config/config.yml.
"""


# --- Imports ---
import pandas as pd
import joblib
from pathlib import Path
from .loading_helpers import load_config

# --- Load config ---
config, PROJECT_ROOT = load_config()

# --- Constants and Paths ---

# Saved objects path
SAVED_OBJECTS_PATH = PROJECT_ROOT / Path(config["data"]["data_path"]) / Path(config["data"]["saved_objects_path"])

# Create a string of the path for use when saving
SAVE_DIR = str(SAVED_OBJECTS_PATH.resolve())

# Data paths
DATA_DIR_PROCESSED = PROJECT_ROOT / Path(config["data"]["data_path"]) / Path(config["data"]["processed_path"])

# Name saving conventions
MODEL = config["saving"]["model_file_suffix"]
HYBRID = config["saving"]["hybrid_file_suffix"]
DP = config["saving"]["dp_file_suffix"]
LAGS =  config["saving"]["lags_file_suffix"]
DESIGN = config["saving"]["design_file_suffix"]
TARGET = config["saving"]["target_file_suffix"]
LINEAR_KEYS = config["saving"]["linear_keys_preffix"]
NON_LINEAR_KEYS = config["saving"]["non_linear_keys_preffix"]
LAGS_PREFFIX = config["saving"]["lags_preffix"]
HYPERPARAMS_PREFFIX = config["saving"]["hyperparams_preffix"]


# -- Functions ---
def load_ts_data() -> tuple[pd.Series, pd.Series]:
    """Load the time series data.

    Returns:
        tuple[pd.Series, pd.Series]: daily and hourly time series as pandas Series inxeded by datetime objects
    """    

    # Get both the full daily and hourly time series
    df_daily = pd.read_csv(DATA_DIR_PROCESSED / config["saving"]["daily_ts_file_name"])
    df_hourly = pd.read_csv(DATA_DIR_PROCESSED / config["saving"]["hourly_ts_file_name"])

    # Convert dates to datetime objects
    df_daily["pickup_date"] = pd.to_datetime(df_daily["pickup_date"])
    df_hourly["dt"] = pd.to_datetime(df_hourly["dt"])

    

    # To pass the time series through our helper functions they need to be a pandas series indexed by a datetime object:
    ts_hourly = pd.Series(df_hourly["trips"].values, index = df_hourly["dt"])

    ts_daily = pd.Series(df_daily["trips"].values, index = df_daily["pickup_date"])

    return ts_daily, ts_hourly

def split_test_train_sets(ts_daily: pd.Series, ts_hourly: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits the time series into test training sets based on the boundaries specified in config.yml

    Args:
        ts_daily (pd.Series): daily time series data
        ts_hourly (pd.Series): hourly time series data

    Returns:
        tuple[pd.Series, pd.Series, pd.Series, pd.Series]: training and testing sets for daily and hourly time series
    """    
    
    # We now need to split into test and train data, we will train on the pre 2024 data and test on 2024 onwards, approx a 90:10 split
    ts_daily_train = ts_daily[:config["modelling"]["ts_daily_train_boundary"]]
    ts_daily_test = ts_daily[config["modelling"]["ts_daily_test_boundary"]:]

    ts_hourly_train = ts_hourly[:config["modelling"]["ts_hourly_train_boundary"]]
    ts_hourly_test = ts_hourly[config["modelling"]["ts_hourly_test_boundary"]:]

    return ts_daily_train, ts_daily_test, ts_hourly_train, ts_hourly_test

def load_process_lags() -> tuple[list[int], list[int]]:
    """Load and process lag features for the time series data.

    Returns:
        tuple[list[int], list[int]]: lists of lag features for daily and hourly time series
    """     
    # First reload the significant lags
    daily_lags = load_lags(config["saving"]["daily_lags"], config["saving"]["lags_sig"])

    hourly_lags = load_lags(config["saving"]["hourly_lags"], config["saving"]["lags_sig"])

    used_hourly_lags = hourly_lags[:config["modelling"]["hourly_num_lags"]]

    # Add any extra lags specified in the config file
    extra_lags = config["modelling"]["hourly_extra_lags"]

    for lag in extra_lags:
        if lag not in used_hourly_lags:
            used_hourly_lags.append(lag)

    return daily_lags, used_hourly_lags

def save_models(linear_models: dict, non_linear_models: dict, sig: str) -> None:
    """ Saves the trained models as pkl files in the saved objects path

    Args:
        linear_models (dict): dictionary containing linear model, deterministic process, hybrid model (or None if purely linear) and lags
        non_linear_models (dict): dictionary containing non linear model, deterministic process and hybrid model (or None if purely non linear)
        sig (str): unique signature to the file names
    """    

    # Note that although we call them linear models, they can also be hybrid models as well the dict structure is the same
    # linear models will just have None for the hybrid component if purely linear

    # Save models
    for key, value in linear_models.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")  
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")
        joblib.dump(value[3], SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")  


    for key, value in non_linear_models.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")
        joblib.dump(value[3], SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")

    # Save the keys so we can reload easily
    joblib.dump(list(linear_models.keys()), SAVED_OBJECTS_PATH / f"{LINEAR_KEYS}_{sig}_{MODEL}") 
    joblib.dump(list(non_linear_models.keys()), SAVED_OBJECTS_PATH / f"{NON_LINEAR_KEYS}_{sig}_{MODEL}") 

def save_design(linear_design: dict, non_linear_design: dict, sig: str) -> None:
    """ Saves the design, target and deterministic process used to train the models

    Args:
        linear_design (dict): dictionary containing design, target and deterministic process for each linear model
        non_linear_design (dict): dictionary containing design, target and deterministic process for each non linear model
        sig (str): unique signature to the file names
    """    
    # Save design, target and deterministic process matricies
    for key, value in linear_design.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        joblib.dump(value[3], SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")

    for key, value in non_linear_design.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        joblib.dump(value[3], SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")

    # Save the keys so we can reload easily
    joblib.dump(list(linear_design.keys()), SAVED_OBJECTS_PATH / f"{LINEAR_KEYS}_{sig}_{DESIGN}") # same keys for linear design
    joblib.dump(list(non_linear_design.keys()), SAVED_OBJECTS_PATH / f"{NON_LINEAR_KEYS}_{sig}_{DESIGN}") # same keys for non linear design


def load_models(sig: str) -> tuple[dict, dict]:
    """ Loads the linear and non linear models stored under the signature sig

    Args:
        sig (str): unique signature to the file names

    Returns:
        Tuple[dict, dict]: dictionaries containing the loaded linear and non linear models
    """    

    # Load keys
    linear_keys = joblib.load(SAVED_OBJECTS_PATH / f"{LINEAR_KEYS}_{sig}_{MODEL}")
    non_linear_keys = joblib.load(SAVED_OBJECTS_PATH / f"{NON_LINEAR_KEYS}_{sig}_{MODEL}")

    # Store models into dicts to return
    non_linear_models_loaded = {}
    linear_models_loaded = {}

    # Loop through both sets of keys and load all components
    for key in linear_keys:
        model = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")
        dp = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        hybrid = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")
        lags = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")
        linear_models_loaded[key] = (model, dp, hybrid, lags)

    for key in non_linear_keys:
        model = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")
        dp = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        hybrid = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")
        lags = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")
        non_linear_models_loaded[key] = (model, dp, hybrid, lags)


    return linear_models_loaded, non_linear_models_loaded 

def load_design(sig: str) -> tuple[dict, dict]:
    """ Loads the design, target and deterministic process stored under the signature sig

    Args:
        sig (str): unique signature to the file names

    Returns:
        Tuple[dict, dict]: dictionaries containing the loaded linear and non linear design, target and deterministic process
    """    

    # Load keys
    linear_keys = joblib.load(SAVED_OBJECTS_PATH / f"{LINEAR_KEYS}_{sig}_{DESIGN}")
    non_linear_keys = joblib.load(SAVED_OBJECTS_PATH / f"{NON_LINEAR_KEYS}_{sig}_{DESIGN}")

    # Create dicts to store design, target and deterministic process to load
    non_linear_design_loaded = {}
    linear_design_loaded = {}

    # Loop through both sets of keys and load all components
    for key in linear_keys:
        X = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        y = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        dp = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        lags = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")
        linear_design_loaded[key] = (X, y, dp, lags)

    for key in non_linear_keys:
        X = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        y = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        dp = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        lags = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{LAGS}")
        non_linear_design_loaded[key] = (X, y, dp, lags)

    return linear_design_loaded, non_linear_design_loaded

def save_lags(lags: list, series_type: str, sig: str) -> None:
    """ Saves the lags used in the models as a pkl file in the saved objects path

    Args:
        lags (list): list of lags to save
        series_type (str): type of the time series (e.g. "hourly", "daily")
        sig (str): unique signature to the file names
    """    
    joblib.dump(lags, SAVED_OBJECTS_PATH / f"{LAGS_PREFFIX}_{series_type}_{sig}.pkl")


def load_lags(series_type: str, sig: str) -> list[int]:
    """ Loads the lags used in the models from a pkl file in the saved objects path

    Args:
        series_type (str): type of the time series (e.g. "hourly", "daily")
        sig (str): unique signature to the file names

    Returns:
        list[int]: list of lags
    """    

    return joblib.load(SAVED_OBJECTS_PATH / f"{LAGS_PREFFIX}_{series_type}_{sig}.pkl")


def save_hyperparams(hyperparams: dict, sig: str) -> None:
    """ Saves the hyperparameters to a pkl file in the saved objects path

    Args:
        hyperparams (dict): dictionary containing the hyperparameters
        sig (str): unique signature to the file names
    """    
    
    joblib.dump(hyperparams, SAVED_OBJECTS_PATH / f"{HYPERPARAMS_PREFFIX}_{sig}.pkl")

def load_hyperparams(sig: str) -> dict:
    """ Loads the hyperparameters from a pkl file in the saved objects path

    Args:
        sig (str): unique signature to the file names

    Returns:
        dict: dictionary containing the hyperparameters
    """    

    return joblib.load(SAVED_OBJECTS_PATH / f"{HYPERPARAMS_PREFFIX}_{sig}.pkl")

def save_obj(obj: object, sig: str) -> None:
    """ Saves an object to a pkl file in the saved objects path

    Args:
        obj (object): object to save
        sig (str): unique signature to the file names
    """    
    joblib.dump(obj, SAVED_OBJECTS_PATH / f"{sig}.pkl")

def load_obj(sig: str) -> object:
    """ Loads an object from a pkl file in the saved objects path

    Args:
        sig (str): unique signature to the file names

    Returns:
        object: loaded object
    """    
    return joblib.load(SAVED_OBJECTS_PATH / f"{sig}.pkl")
