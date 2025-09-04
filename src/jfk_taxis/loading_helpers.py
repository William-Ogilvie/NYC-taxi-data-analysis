import yaml
from pathlib import Path

# Function to load and return config, will also return the project root
def load_config():
    # src/jfk_taxis is location of current file so we go two above to get project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Path to config
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config, PROJECT_ROOT