"""
test_loading_helpers.py
=========================

Unit tests for loading_helpers.py
"""

def test_load_config():
    """ Test the load_config function to ensure it returns a dictionary and a Path object.
    """
    from jfk_taxis import loading_helpers
    from pathlib import Path

    config, project_root = loading_helpers.load_config()

    assert isinstance(config, dict), "Config should be a dictionary"
    assert isinstance(project_root, Path), "Project root should be a Path object"

def test_save_config():
    """ Test the save_config function to ensure it saves the config to a file.
    """    
    from jfk_taxis import loading_helpers

    # Load the original config
    config, PROJECT_ROOT = loading_helpers.load_config()

    # Modify the config slightly
    config["test_key"] = "test_value"

    # Save the modified config
    config, PROJECT_ROOT = loading_helpers.save_config(config)

    # Check that the modification is present
    assert config.get("test_key") == "test_value", "Config modification not saved"

    # Clean up by removing the test key
    del config["test_key"]

    # Save the cleaned config
    config, PROJECT_ROOT = loading_helpers.save_config(config)

    assert "test_key" not in config, "Config cleanup failed"