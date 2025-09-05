import yaml
from pathlib import Path

def load_config() -> tuple[dict, Path]:
    """ Loads the configuration from a YAML file and determines the project root path. Returns both.

    Returns:
        tuple[dict, Path]: A tuple containing the config dictionary and the project root path.
    """    


    # src/jfk_taxis is location of current file so we go two above to get project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    # Path to config
    CONFIG_PATH = PROJECT_ROOT / "config" / "config.yml"

    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config, PROJECT_ROOT