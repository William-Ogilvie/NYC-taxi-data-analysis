import joblib
from pathlib import Path

# src/jfk_taxis/ is location of current file so we go two above to get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Save dir
SAVE_DIR_LOC = PROJECT_ROOT / "data" / "saved_objects"

# Create the directory if it doesn't already exist
SAVE_DIR_LOC.mkdir(parents= True, exist_ok= True)

# Create a string of the path for use when saving
SAVE_DIR = str(SAVE_DIR_LOC.resolve())

 

# Function saves the design and target matrices, the models themselves and their deterministic processes into .pkl objects using joblib
def save_models(linear_models: dict, non_linear_models: dict, sig: str):
    """
    linear_models: dict where 0: is the linear model, 1: is the determinsitic process, 2: is the hybrid component (so for boosted residuals this is just an XGBoost)
    non_linear_models: dict where 0: non linear model (XGBoost), 1: deterministic process, 2: hybrid component if any 
    sig: str this is a unique signature to the file names to avoid saving to things to the same file, for example for daily hyrbid models use something like hybrid_daily 
    """

    # Note that although we call them linear models, they can also be hybrid models as well the dict structure is the same
    # linear models will just have None for the hybrid component if purely linear

    # Save models
    for key, value in linear_models.items():
        joblib.dump(value[0], f"{SAVE_DIR}/{key}_{sig}_model.pkl")
        joblib.dump(value[1], f"{SAVE_DIR}/{key}_{sig}_dp.pkl")
        joblib.dump(value[2], f"{SAVE_DIR}/{key}_{sig}_hybrid.pkl")


    for key, value in non_linear_models.items():
        joblib.dump(value[0], f"{SAVE_DIR}/{key}_{sig}_model.pkl")
        joblib.dump(value[1], f"{SAVE_DIR}/{key}_{sig}_dp.pkl")
        joblib.dump(value[2], f"{SAVE_DIR}/{key}_{sig}_hybrid.pkl")


    # Save the keys so we can reload easily
    joblib.dump(list(linear_models.keys()), f"{SAVE_DIR}/Linear_keys_{sig}_models.pkl") 
    joblib.dump(list(non_linear_models.keys()), f"{SAVE_DIR}/Non_linear_keys_{sig}_models.pkl") 

# Function saves the design, target and deterministic process used to train the models
def save_design(linear_design: dict, non_linear_design: dict, sig: str):
    """
    linear_design: dict of design matricies for the model 0: is design matrix, 1: is target vector, 2: is deterministic process   
    non_linear_design: dict where 0: design matrix, 1: target vector, 2: deterministic process
    sig: str this is a unique signature to the file names to avoid saving to things to the same file, for example for daily hyrbid models use something like hybrid_daily  
    """ 
    
    # Save design, target and deterministic process matricies
    for key, value in linear_design.items():
        joblib.dump(value[0], f"{SAVE_DIR}/{key}_{sig}_design.pkl")
        joblib.dump(value[1], f"{SAVE_DIR}/{key}_{sig}_target.pkl")
        joblib.dump(value[2], f"{SAVE_DIR}/{key}_{sig}_dp1.pkl")

    for key, value in non_linear_design.items():
        joblib.dump(value[0], f"{SAVE_DIR}/{key}_{sig}_design.pkl")
        joblib.dump(value[1], f"{SAVE_DIR}/{key}_{sig}_target.pkl")
        joblib.dump(value[2], f"{SAVE_DIR}/{key}_{sig}_dp1.pkl")

    # Save the keys so we can reload easily
    joblib.dump(list(linear_design.keys()), f"{SAVE_DIR}/Linear_keys_{sig}_design.pkl") # same keys for linear design
    joblib.dump(list(non_linear_design.keys()), f"{SAVE_DIR}/Non_linear_keys_{sig}_design.pkl") # same keys for non linear design


# Loads the linear and non linear models stored under the signature sig
def load_models(sig: str):
    # Load keys
    linear_keys = joblib.load(f"{SAVE_DIR}/Linear_keys_{sig}_models.pkl")
    non_linear_keys = joblib.load(f"{SAVE_DIR}/Non_linear_keys_{sig}_models.pkl")

    # Store models into dicts to return
    non_linear_models_loaded = {}
    linear_models_loaded = {}

    # Loop through both sets of keys and load all components
    for key in linear_keys:
        model = joblib.load(f"{SAVE_DIR}/{key}_{sig}_model.pkl")
        dp = joblib.load(f"{SAVE_DIR}/{key}_{sig}_dp.pkl")
        hybrid = joblib.load(f"{SAVE_DIR}/{key}_{sig}_hybrid.pkl")
        linear_models_loaded[key] = (model, dp, hybrid)

    for key in non_linear_keys:
        model = joblib.load(f"{SAVE_DIR}/{key}_{sig}_model.pkl")
        dp = joblib.load(f"{SAVE_DIR}/{key}_{sig}_dp.pkl")
        hybrid = joblib.load(f"{SAVE_DIR}/{key}_{sig}_hybrid.pkl")
        non_linear_models_loaded[key] = (model, dp, hybrid)


    return linear_models_loaded, non_linear_models_loaded 

def load_design(sig: str):

    # Load keys
    linear_keys = joblib.load(f"{SAVE_DIR}/Linear_keys_{sig}_design.pkl")
    non_linear_keys = joblib.load(f"{SAVE_DIR}/Non_linear_keys_{sig}_design.pkl")

    # Create dicts to store design, target and deterministic process to load
    non_linear_design_loaded = {}
    linear_design_loaded = {}

    # Loop through both sets of keys and load all components
    for key in linear_keys:
        X = joblib.load(f"{SAVE_DIR}/{key}_{sig}_design.pkl")
        y = joblib.load(f"{SAVE_DIR}/{key}_{sig}_target.pkl")
        dp = joblib.load(f"{SAVE_DIR}/{key}_{sig}_dp1.pkl")
        linear_design_loaded[key] = (X, y, dp)

    for key in non_linear_keys:
        X = joblib.load(f"{SAVE_DIR}/{key}_{sig}_design.pkl")
        y = joblib.load(f"{SAVE_DIR}/{key}_{sig}_target.pkl")
        dp = joblib.load(f"{SAVE_DIR}/{key}_{sig}_dp1.pkl")
        non_linear_design_loaded[key] = (X, y, dp)

    return linear_design_loaded, non_linear_design_loaded

# Function to save lags, exists primarily to ensure that everything ends up in the same folder at run time
def save_lags(lags: list, series_type: str, sig: str):
    joblib.dump(lags, f"{SAVE_DIR}/sig_lags_{series_type}_{sig}.pkl")

# Function to load lags
def load_lags(series_type: str, sig: str):
    return joblib.load(f"{SAVE_DIR}/sig_lags_{series_type}_{sig}.pkl")
    



