import joblib
from pathlib import Path
import yaml

# src/jfk_taxis/ is location of current file so we go two above to get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to config
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Saved objects path
SAVED_OBJECTS_PATH = PROJECT_ROOT / Path(config["data"]["data_path"]) / Path(config["data"]["saved_objects_path"])

# Create a string of the path for use when saving
SAVE_DIR = str(SAVED_OBJECTS_PATH.resolve())

# Name saving conventions
MODEL = config["saving"]["model_file_suffix"]
HYBRID = config["saving"]["hybrid_file_suffix"]
DP = config["saving"]["dp_file_suffix"]
DESIGN = config["saving"]["design_file_suffix"]
TARGET = config["saving"]["target_file_suffix"]
LINEAR_KEYS = config["saving"]["linear_keys_preffix"]
NON_LINEAR_KEYS = config["saving"]["non_linear_keys_preffix"]
LAGS_PREFFIX = config["saving"]["lags_preffix"]
HYPERPARAMS_PREFFIX = config["saving"]["hyperparams_preffix"]

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
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")  
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")


    for key, value in non_linear_models.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")


    # Save the keys so we can reload easily
    joblib.dump(list(linear_models.keys()), SAVED_OBJECTS_PATH / f"{LINEAR_KEYS}_{sig}_{MODEL}") 
    joblib.dump(list(non_linear_models.keys()), SAVED_OBJECTS_PATH / f"{NON_LINEAR_KEYS}_{sig}_{MODEL}") 

# Function saves the design, target and deterministic process used to train the models
def save_design(linear_design: dict, non_linear_design: dict, sig: str):
    """
    linear_design: dict of design matricies for the model 0: is design matrix, 1: is target vector, 2: is deterministic process   
    non_linear_design: dict where 0: design matrix, 1: target vector, 2: deterministic process
    sig: str this is a unique signature to the file names to avoid saving to things to the same file, for example for daily hyrbid models use something like hybrid_daily  
    """ 
    
    # Save design, target and deterministic process matricies
    for key, value in linear_design.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")

    for key, value in non_linear_design.items():
        joblib.dump(value[0], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        joblib.dump(value[1], SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        joblib.dump(value[2], SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")

    # Save the keys so we can reload easily
    joblib.dump(list(linear_design.keys()), SAVED_OBJECTS_PATH / f"{LINEAR_KEYS}_{sig}_{DESIGN}") # same keys for linear design
    joblib.dump(list(non_linear_design.keys()), SAVED_OBJECTS_PATH / f"{NON_LINEAR_KEYS}_{sig}_{DESIGN}") # same keys for non linear design


# Loads the linear and non linear models stored under the signature sig
def load_models(sig: str):
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
        linear_models_loaded[key] = (model, dp, hybrid)

    for key in non_linear_keys:
        model = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{MODEL}")
        dp = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        hybrid = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{HYBRID}")
        non_linear_models_loaded[key] = (model, dp, hybrid)


    return linear_models_loaded, non_linear_models_loaded 

def load_design(sig: str):

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
        linear_design_loaded[key] = (X, y, dp)

    for key in non_linear_keys:
        X = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DESIGN}")
        y = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{TARGET}")
        dp = joblib.load(SAVED_OBJECTS_PATH / f"{key}_{sig}_{DP}")
        non_linear_design_loaded[key] = (X, y, dp)

    return linear_design_loaded, non_linear_design_loaded

# Function to save lags, exists primarily to ensure that everything ends up in the same folder at run time
def save_lags(lags: list, series_type: str, sig: str):
    joblib.dump(lags, SAVED_OBJECTS_PATH / f"{LAGS_PREFFIX}_{series_type}_{sig}.pkl")

# Function to load lags
def load_lags(series_type: str, sig: str):
    return joblib.load(SAVED_OBJECTS_PATH / f"{LAGS_PREFFIX}_{series_type}_{sig}.pkl")


# Function to save hyperparams
def save_hyperparams(hyperparams: dict, sig: str):
    joblib.dump(hyperparams, SAVED_OBJECTS_PATH / f"{HYPERPARAMS_PREFFIX}_{sig}.pkl")

# Function to load hyperparams
def load_hyperparams(sig: str):
    return joblib.load(SAVED_OBJECTS_PATH / f"{HYPERPARAMS_PREFFIX}_{sig}.pkl")

# Function to save any object
def save_obj(obj, sig: str):
    joblib.dump(obj, SAVED_OBJECTS_PATH / f"{sig}.pkl")

# Function to load any object
def load_obj(sig: str):
    return joblib.load(SAVED_OBJECTS_PATH / f"{sig}.pkl")
