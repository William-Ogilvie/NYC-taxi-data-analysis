"""
loading_helpers
=================

This module contains just the load_config function which is used to load the config from config/config.yml and get the project root path.
"""


# --- Imports ---
import yaml
from pathlib import Path

# --- Constants ---

# src/jfk_taxis is location of current file so we go two above to get project root 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

# --- Functions ---
def load_config() -> tuple[dict, Path]:
    """ Loads the configuration from a YAML file and determines the project root path. Returns both.

    Returns:
        tuple[dict, Path]: A tuple containing the config dictionary and the project root path.
    """    
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config, PROJECT_ROOT

def save_config(config: dict) -> None:
    """ Saves the configuration dictionary back to the YAML file.

    Args:
        config (dict): The configuration dictionary to save.
    """
    with open(PROJECT_ROOT / "config" / "config.yml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style= True)
