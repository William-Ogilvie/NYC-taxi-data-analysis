"""
setup
=============

This script will create all the directories we need for the project as specified in config/config.yml.
This script should be run before get_parquet.py to ensure the data/raw directory exists before trying to
download the data.
"""

# --- Imports ---
from pathlib import Path
import xgboost
import yaml
import cupy as cp
import numpy as np

# scripts/ is location of current file so we go one above to get project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to config
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

# Load config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# We are now going to make all the directories we need for the project if they don't already exist

# Data directories
raw_data_dir = PROJECT_ROOT / Path(config["data"]["data_path"]) / Path(config["data"]["raw_path"])
processed_data_dir = PROJECT_ROOT / Path(config["data"]["data_path"]) / Path(config["data"]["processed_path"])
saved_objects_dir = PROJECT_ROOT / Path(config["data"]["data_path"]) / Path(config["data"]["saved_objects_path"])
figures_dir = PROJECT_ROOT / Path(config["data"]["reports_path"]) / Path(config["data"]["figures_path"])
maps_dir = PROJECT_ROOT / Path(config["data"]["reports_path"]) / Path(config["data"]["maps_path"])

# Create the directories if they don't exist already exist
raw_data_dir.mkdir(parents= True, exist_ok= True)
processed_data_dir.mkdir(parents= True, exist_ok= True)
saved_objects_dir.mkdir(parents= True, exist_ok= True)
figures_dir.mkdir(parents= True, exist_ok= True)
maps_dir.mkdir(parents= True, exist_ok= True)

# --- Functions ---
def test_gpu():
    """Test GPU availability for XGBoost.

    Returns:
        str: "cuda" if GPU is available, "cpu" otherwise.
    """    
    try:
        X = cp.array([[0.0], [1.0]], dtype=np.float32)
        y = cp.array([0.0, 1.0], dtype=np.float32) 

        model = xgboost.XGBRegressor(
            n_estimators = 1,
            max_depth = 1,
            tree_method = "hist",
            device = "cuda",
            verbosity = 0,
        )
        model.fit(X, y)

        return "cuda"
    except Exception as e:
        print("GPU not available, falling back to CPU.")

        return "cpu"

# --- Main ---
if __name__ == "__main__":

    # Test if GPU is available for XGBoost and update config
    device = test_gpu()

    config["xgboost_setup"]["device"] = device

    # Save updated config
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, default_flow_style= True)



