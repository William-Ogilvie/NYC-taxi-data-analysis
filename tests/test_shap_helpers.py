"""
test_shap_helpers.py
=========================

Unit tests for shap_helpers.py
"""   

def create_models_designs_for_shap_tests() -> tuple[str, str, str, str]:
    """ This function creates models and designs to be used in the shap tests. It uses the same setup as in the modelling_helpers tests.
    
    Returns:
        tuple[str, str, str, str]: The design and model signatures for daily linear, hourly linear, daily hybrid and hourly hybrid models. 
    """    
    from .test_modelling_helpers import default_xgb_model, expected_xgbregressor_params, create_ts
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



def test_compute_shap_values():
    """ test the compute_shap_values function to ensure it returns SHAP values and a design matrix.
    """    
    from jfk_taxis import shap_helpers

    # This function is going to expect a design sig and a model sig.
    # We will reuse the ones from the modelling_helpers tests, this is implemented in the create_models_designs_for_shap_tests above
    daily_linear_sig, hourly_linear_sig, daily_hybrid_sig, hourly_hybrid_sig = create_models_designs_for_shap_tests()



