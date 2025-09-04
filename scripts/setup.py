from pathlib import Path
import yaml

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



