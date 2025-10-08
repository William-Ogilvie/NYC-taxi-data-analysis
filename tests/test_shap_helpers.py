"""
test_shap_helpers.py
=========================

Unit tests for shap_helpers.py
"""   
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import shap
import pandas as pd

def create_models_designs_for_shap_tests() -> tuple[str, str, str, str]:
    """ This function creates models and designs to be used in the shap tests. It uses the same setup as in the modelling_helpers tests.
    
    Returns:
        tuple[str, str, str, str]: The design and model signatures for daily linear, hourly linear, daily hybrid and hourly hybrid models. 
    """    
    from test_modelling_helpers import default_xgb_model, expected_xgbregressor_params, create_ts
    from jfk_taxis import modelling_helpers
    import copy

    daily_lags = [1, 7, 14, 30, 60]
    hourly_lags = [1, 24, 168, 336, 720]
    daily_fourier_features = ["YE", "W"]
    hourly_fourier_features = ["D", "W"]

    expected_daily_lags = [f"y_lag_{lag}" for lag in daily_lags]
    expected_hourly_lags = [f"y_lag_{lag}" for lag in hourly_lags]

    expected_daily_fourier = [f"cos({x},freq=YE-DEC)" for x in range(1, 11)] + [f"sin({x},freq=YE-DEC)" for x in range(1, 11)] + \
                            [f"cos({x},freq=W-SUN)" for x in range(1, 6)] + [f"sin({x},freq=W-SUN)" for x in range(1, 6)]
    expected_hourly_fourier = [f"cos({x},freq=D)" for x in range(1, 6)] + [f"sin({x},freq=D)" for x in range(1, 6)] + \
                            [f"cos({x},freq=W-SUN)" for x in range(1, 6)] + [f"sin({x},freq=W-SUN)" for x in range(1, 6)]

    # First model is order 3 second order 2 
    order_list = [3, 2]
    expected_3_order_terms = ["const", "trend", "trend_squared", "trend_cubed"]
    expected_2_order_terms = ["const", "trend", "trend_squared"]

    # XGBRegressor to use
    xgbregressor = default_xgb_model()

    # Expected params for this model
    expected_params = expected_xgbregressor_params()

    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    daily_names_linear = ["model_1", "model_2"]
    hourly_names_linear = ["model_2", "model_3"]

    daily_names_non_linear = ["model_4", "model_5"]
    hourly_names_non_linear = ["model_5", "model_6"]

    daily_names_hybrid = ["model_7", "model_8"]
    hourly_names_hybrid = ["model_8", "model_9"]

    # We will run the function twice per time step once for linear, once for hybrid
    # To reload the models for checking we are going to have to use load_models and load_designs from training_helpers 
    daily_linear_sig = "linear_daily"
    hourly_linear_sig = "linear_hourly"

    daily_hybrid_sig = "hybrid_daily"
    hourly_hybrid_sig = "hybrid_hourly"

    # Daily linear
    modelling_helpers.create_train_save_models(daily_names_linear, daily_names_non_linear, None, daily_linear_sig, order_list, daily_lags, daily_fourier_features, "D", daily_ts)

    # Daily hybrid
    hybrid = copy.deepcopy(xgbregressor)
    modelling_helpers.create_train_save_models(daily_names_hybrid, daily_names_non_linear, hybrid, daily_hybrid_sig, order_list, daily_lags, daily_fourier_features, "D", daily_ts)

    # Hourly linear
    modelling_helpers.create_train_save_models(hourly_names_linear, hourly_names_non_linear, None, hourly_linear_sig, order_list, hourly_lags, hourly_fourier_features, "h", hourly_ts)

    # Hourly hybrid
    hybrid = copy.deepcopy(xgbregressor)
    modelling_helpers.create_train_save_models(hourly_names_hybrid, hourly_names_non_linear, hybrid, hourly_hybrid_sig, order_list, hourly_lags, hourly_fourier_features, "h", hourly_ts)

    return daily_linear_sig, hourly_linear_sig, daily_hybrid_sig, hourly_hybrid_sig

def get_shap_values(model: LinearRegression | XGBRegressor, X: pd.DataFrame) -> shap.Explanation:
    """ gets shap values for a given model and design matrix.

    Args:
        model (LinearRegression | XGBRegressor): the model
        X (pd.DataFrame): the design matrix

    Returns:
        shap.Explanation: the SHAP values
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values


def test_compute_shap_values():
    """ test the compute_shap_values function to ensure it returns SHAP values and a design matrix.
    """    
    from jfk_taxis import shap_helpers
    from jfk_taxis import load_design, load_models 
    import numpy as np

    # This function is going to expect a design sig and a model sig.
    # We will reuse the ones from the modelling_helpers tests, this is implemented in the create_models_designs_for_shap_tests above
    daily_linear_sig, hourly_linear_sig, daily_hybrid_sig, hourly_hybrid_sig = create_models_designs_for_shap_tests()

    # Daily linear
    # We will look at "model_1" which is a linear model and "model4" which is a non-linear model
    shap_values, X = shap_helpers.compute_shap_values(daily_linear_sig, daily_linear_sig, "model_1", linear=True, hybrid=False)

    # We will use load_design and load_model to check the outputs
    linear_design_loaded, non_linear_design_loaded = load_design(daily_linear_sig)
    linear_model_loaded, non_linear_model_loaded = load_models(daily_linear_sig)

    X_check = linear_design_loaded["model_1"][0]
    model = linear_model_loaded["model_1"][0]

    shap_values_check = get_shap_values(model, X)

    print(shap_values_check)

    # Check that we have a match, rtol is relative tolerance, atol is absolute tolerance
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match" 
    assert X.equals(X_check), "Design matrices do not match"

    # Repeat for non linear model
    shap_values, X = shap_helpers.compute_shap_values(daily_linear_sig, daily_linear_sig, "model_4", linear=False, hybrid=False)

    X_check = non_linear_design_loaded["model_4"][0]
    model = non_linear_model_loaded["model_4"][0]

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match"  
    assert X.equals(X_check), "Design matrices do not match"

    # Now repeat for hybrid daily, we use "model_7" which is the hybrid model and "model_5" which is the non-linear model
    shap_values, X = shap_helpers.compute_shap_values(daily_hybrid_sig, daily_hybrid_sig, "model_7", linear=True, hybrid=True)

    linear_design_loaded, non_linear_design_loaded = load_design(daily_hybrid_sig)
    linear_model_loaded, non_linear_model_loaded = load_models(daily_hybrid_sig)

    X_check = linear_design_loaded["model_7"][0]
    model = linear_model_loaded["model_7"][2]  # Hybrid model is index 2

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match"  
    assert X.equals(X_check), "Design matrices do not match"

    # Repeat for non linear model
    shap_values, X = shap_helpers.compute_shap_values(daily_hybrid_sig, daily_hybrid_sig, "model_5", linear=False, hybrid=False)

    X_check = non_linear_design_loaded["model_5"][0]
    model = non_linear_model_loaded["model_5"][0]

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match" 
    assert X.equals(X_check), "Design matrices do not match"

    # Repeat for hourly linear, we use "model_2" which is a linear model and "model_5" which is a non-linear model
    shap_values, X = shap_helpers.compute_shap_values(hourly_linear_sig, hourly_linear_sig, "model_2", linear=True, hybrid=False)

    linear_design_loaded, non_linear_design_loaded = load_design(hourly_linear_sig) 
    linear_model_loaded, non_linear_model_loaded = load_models(hourly_linear_sig)

    X_check = linear_design_loaded["model_2"][0]
    model = linear_model_loaded["model_2"][0]

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match"    
    assert X.equals(X_check), "Design matrices do not match"

    # Repeat for non linear model
    shap_values, X = shap_helpers.compute_shap_values(hourly_linear_sig, hourly_linear_sig, "model_5", linear=False, hybrid=False)
    X_check = non_linear_design_loaded["model_5"][0]
    model = non_linear_model_loaded["model_5"][0]

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match"    
    assert X.equals(X_check), "Design matrices do not match"

    # Repeat for hybrid hourly we use "model_8" which is the hybrid model and "model_6" which is the non-linear model
    shap_values, X = shap_helpers.compute_shap_values(hourly_hybrid_sig, hourly_hybrid_sig, "model_8", linear=True, hybrid=True)

    linear_design_loaded, non_linear_design_loaded = load_design(hourly_hybrid_sig)
    linear_model_loaded, non_linear_model_loaded = load_models(hourly_hybrid_sig)

    X_check = linear_design_loaded["model_8"][0]
    model = linear_model_loaded["model_8"][2]  # Hybrid model is index

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"   
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match"
    assert X.equals(X_check), "Design matrices do not match"

    # Repeat for non linear model
    shap_values, X = shap_helpers.compute_shap_values(hourly_hybrid_sig, hourly_hybrid_sig, "model_6", linear=False, hybrid=False)

    X_check = non_linear_design_loaded["model_6"][0]
    model = non_linear_model_loaded["model_6"][0]

    shap_values_check = get_shap_values(model, X)

    # Check that we have a match
    np.testing.assert_allclose(shap_values_check.values, shap_values.values, rtol = 1e-5, atol = 1e-5), "SHAP values do not match"
    np.testing.assert_allclose(shap_values_check.base_values, shap_values.base_values, rtol = 1e-5, atol = 1e-5), "SHAP base values do not match"
    np.testing.assert_allclose(shap_values_check.data, shap_values.data, rtol = 1e-5, atol = 1e-5), "SHAP data do not match"  
    assert X.equals(X_check), "Design matrices do not match"

def test_shap_plots():
    """ test the shap_plots function to ensure it runs without errors.
    
    This is a smoke test - we just verify the function runs successfully with valid inputs.
    """
    from jfk_taxis import shap_helpers
    from jfk_taxis import load_design, load_models
    import matplotlib.pyplot as plt
    import numpy as np

    # Use non-interactive backend and stub show
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Create models and designs for testing
        daily_linear_sig, hourly_linear_sig, daily_hybrid_sig, hourly_hybrid_sig = create_models_designs_for_shap_tests()

        # Load a model and design to get SHAP values
        shap_values, X = shap_helpers.compute_shap_values(daily_linear_sig, daily_linear_sig, "model_1", linear=True, hybrid=False)

        # Run shap_plots - should not raise any errors
        shap_helpers.shap_plots(shap_values, X, "model_1")

        # Close plots to avoid warnings
        plt.close("all")

        # Test with a non-linear model as well
        shap_values, X = shap_helpers.compute_shap_values(daily_linear_sig, daily_linear_sig, "model_4", linear=False, hybrid=False)
        shap_helpers.shap_plots(shap_values, X, "model_4")

        # Close plots to avoid warnings
        plt.close("all")

    finally:
        plt.show = old_show


def test_return_top_X_SHAP():
    """ test the return_top_X_SHAP function to ensure it returns the top X SHAP values.
    """    
    from jfk_taxis import shap_helpers
    import numpy as np

    # Create dummy SHAP values and features names, now feature 1 should have lowest mean, feature 3 highest mean
    shap_values = np.array([
        [1, 2, 3],
        [0, 2, 4],
        [1, 3, 5]
    ])

    feature_names = pd.Index(["feature_1", "feature_2", "feature_3"])

    top_2 = shap_helpers.return_top_X_SHAP(shap_values, feature_names, 2)

    expected_top_2 = [("feature_3", 4.0), ("feature_2", 2.3333333333333335)]

    assert top_2 == expected_top_2, f"Expected {expected_top_2}, but got {top_2}"

def test_extract_features_from_top_x():
    """ tests the extract_features_from_top_x function 
    """    
    from jfk_taxis import shap_helpers

    # Create a dummy top_x list
    top_x = [("lag_7", 0.5), ("cos(1,freq=YE-DEC)", 0.4), ("trend", 0.3), ("lag_1", 0.25), ("sin(5,freq=W-SUN)", 0.2), ("cos(3,freq=D)", 0.15), ("trend_squared", 0.1), ("const", 0.05)]

    lags, fourier_features, trends = shap_helpers.extract_features_from_top_x(top_x)

    assert set(lags) == {1, 7}, f"Expected lags {1, 7}, but got {lags}"
    assert set(fourier_features) == {"YE", "W", "D"}, f"Expected fourier features {'YE', 'W', 'D'}, but got {fourier_features}"
    assert set(trends) == {"trend", "trend_squared", "const"}, f"Expected trends {'trend', 'trend_squared', 'const'}, but got {trends}"

class FakeSHAPExplainer:
    """ fake shap.Explainer class to pass through our function
    """    
    def __init__(self, values):
        self.values = values


def test_extract_top_x_features_dict():
    """ tests the extract_top_x_features_dict function.
    """
    import numpy as np    
    from jfk_taxis import shap_helpers

    # We will create three sets of shap values to test

    # lag_1 lowest, sin(1,freq=YE-DEC) highest, trend 2nd highest
    shap_values_1 = np.array([    
        [1, 2, 3],
        [0, 2, 4],
        [1, 3, 5]
    ])

    # Create a fake shap explainer object to pass through function
    shap_explainer_1 = FakeSHAPExplainer(shap_values_1)

    X_1 = pd.DataFrame({
        "lag_1": [0.1, 0.2, 0.3],
        "trend": [0.3, 0.4, 0.1],
        "sin(1,freq=YE-DEC)": [0.01, 0.3, 0.4]
    })
    

    # cos(3,freq=D) highest then lag_6, then lag_12, then sin(4,freq=W-SUN), this model allows us to test that the lags are returned in ascending order
    shap_values_2 = np.array([
        [3, 0, 1, 1],
        [4, 0, 2, 2],
        [5, 1, 3, 2]
    ])

    # Create a fake shap explainer object to pass through function
    shap_explainer_2 = FakeSHAPExplainer(shap_values_2)

    X_2 = pd.DataFrame({
        "cos(3,freq=D)": [0, 1, 2],
        "sin(4,freq=W-SUN)": [0.5, 2, 4],
        "lag_6": [0.2, 3, 4],
        "lag_12": [0.3, 4, 1]
    })

    # sin(4,freq=D) largest, then trend_squared, cos(2,freq=W-SUN), sin(5,freq=YE-DEC), lag_275
    shap_values_3 = np.array([
        [3, 1, 3, 6, 1],
        [3, 0, 5, 8, 2],
        [2, 0, 5, 9, 0],
        [1, 1, 4, 5, 9]
        # 4.5, 1, 8.5, 19, 6
    ])

    # Create a fake shap explainer object to pass through function
    shap_explainer_3 = FakeSHAPExplainer(shap_values_3)

    X_3 = pd.DataFrame({
        "sin(5,freq=YE-DEC)": [0, 1, 2, 3],
        "lag_275": [0, 1, 2, 4],
        "trend_squared": [0.1, 22, 1, 0],
        "sin(4,freq=D)": [0.4, 1, 2, 3],
        "cos(2,freq=W-SUN)": [0.1, 0, 0, 1]
    })

    shap_values_dict = {
        "model_1": (shap_explainer_1, X_1),
        "model_2": (shap_explainer_2, X_2),
        "model_7": (shap_explainer_3, X_3)
    }

    # Extract top 3 features for each model
    features_dict = shap_helpers.extract_top_x_features_dict(shap_values_dict, 3)

    # Checks
    assert features_dict["model_1"][0] == [1], "lags for model_1 don't match"
    assert features_dict["model_1"][1] == ["YE"], "fourier features for model_1 don't match"
    assert features_dict["model_1"][2] == ["trend"], "trend for model_1 doesn't match"
    assert features_dict["model_2"][1] == ["D"], "fourier features for model_2 doesn't match"
    assert features_dict["model_2"][2] == [], "trend for model_2 doesn't match"
    assert features_dict["model_2"][0] == [6, 12], "lags for model_2 doesn't match"
    assert features_dict["model_7"][0] == [], "lags for model_7 don't match"
    assert set(features_dict["model_7"][1]) == {"D", "W"}, "fourier features for model_7 don't match"
    assert features_dict["model_7"][2] == ["trend_squared"], "trend for model_7 doesn't match"

def test_save_extracted_features_to_config():
    """ tests the save_extracted_features_to_config function.
    """
    from jfk_taxis import load_config
    from jfk_taxis import shap_helpers

    # Load config
    config, PROJECT_ROOT = load_config()    

    features_dict = {
        "test_model_1": ([7, 12], [], []),
        "test_model_2": ([2], ["YE", "W"], ["trend"]),
        "test_model_17": ([], ["D"], ["const"])
    }

    # Run the function
    config = shap_helpers.save_extracted_features_to_config(features_dict, config)

    # Reload the config to check that both the config returned is correct and that the config stored in config/config.yml is correct
    config_reloaded, PROJECT_ROOT = load_config()

    # Checks
    assert config == config_reloaded, "Reloaded config doesn't match returned config"
    assert config["shap"]["test_model_1"]["extracted_lags"] == [7, 12], "test_model_1's lags don't match"
    assert config["shap"]["test_model_1"]["extracted_fourier_features"] == [], "test_model_1's fourier features don't match"
    assert config["shap"]["test_model_1"]["extracted_trends"] == [], "test_model_1's trends don't match"
    assert config["shap"]["test_model_2"]["extracted_lags"] == [2], "test_model_2's lags don't match"
    assert config["shap"]["test_model_2"]["extracted_fourier_features"] == ["YE", "W"], "test_model_2's fourier features don't match"
    assert config["shap"]["test_model_2"]["extracted_trends"] == ["trend"], "test_model_2's trends don't match"
    assert config["shap"]["test_model_17"]["extracted_lags"] == [], "test_model_17's lags don't match"
    assert config["shap"]["test_model_17"]["extracted_fourier_features"] == ["D"], "test_model_17's fourier features don't match"
    assert config["shap"]["test_model_17"]["extracted_trends"] == ["const"], "test_model_17's trends don't match"


    
    





