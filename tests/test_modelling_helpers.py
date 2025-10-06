"""
test_modelling_helpers.py
==========================

Unit tests for modelling_helpers.py
"""
import pandas as pd
from xgboost import XGBRegressor

def create_ts(time_step: str) -> pd.Series:
    """ creates time series for testing

    Args:
        time_step (str): time step of series

    Returns:
        pd.Series: time series
    """    
    if time_step == "D":
        return pd.Series(data = range(1, 365*3 + 1), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3, freq = "D"))
    elif time_step == "h":
        return pd.Series(data = range(1, 365*3*24 + 1), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3*24, freq = "h"))

def test_make_offsets():
    """ test the make_offsets function in modelling_helpers.py.
    """    
    from jfk_taxis import modelling_helpers
    import numpy as np

    # Load config
    config, PROJECT_ROOT = modelling_helpers.load_config()

    total_time = 10
    offset_step = 3

    offsets = modelling_helpers.make_offsets(total_time, offset_step)
    
    # Checks
    assert len(offsets) == 4, "Number of offsets not correct"
    assert isinstance(offsets, list), "Offsets not a list"
    assert all(isinstance(i, int) for i in offsets), "Offsets not all integers"
    assert all((offsets[i] >= (3*i - 1)) and offsets[i] <= (3*i + 1) for i in range(0, len(offsets))), "Offsets not in correct ranges" 
    assert offsets[-1] <= total_time, "Last offset exceeds total time"

def test_make_offsets_from_series():
    """ test the make_offsets_from_series function in modelling_helpers.py.
    """
    from jfk_taxis import modelling_helpers
    import pandas as pd

    # Create a daily and hourly series
    series_daily_full =  create_ts("D")
    series_hourly_full = create_ts("h")
    
    # Daily
    total_time = len(series_daily_full)
    offset_step = 30
    forecast_steps = [7, 30, 60]

    offsets_daily = modelling_helpers.make_offsets_from_series(series_daily_full, offset_step, forecast_steps)

    # Checks
    assert len(offsets_daily) == 34, "Number of daily offsets not correct (365*3+1 - 60 - 15)/30 is 34 rounded down"
    assert isinstance(offsets_daily, list), "Daily offsets not a list"
    assert all(isinstance(i, int) for i in offsets_daily), "Daily offsets not all integers" 
    assert all((offsets_daily[i] >= (30*i - 15)) and offsets_daily[i] <= (30*i + 15) for i in range(0, len(offsets_daily))), "Daily offsets not in correct ranges"
    assert all((offsets_daily[-1] + step <= total_time for step in forecast_steps)), "Last daily offset + forecast step exceeds total time"

    # Hourly
    total_time = len(series_hourly_full)
    offset_step = 720
    forecast_steps = [24, 168, 336]

    offsets_hourly = modelling_helpers.make_offsets_from_series(series_hourly_full, offset_step, forecast_steps)

    # Checks
    assert len(offsets_hourly) == 36, "Number of hourly offsets not correct (365*3*24+1 - 336 - 360)/720 is 35 rounded down so 36 offsets"
    assert isinstance(offsets_hourly, list), "Hourly offsets not a list"
    assert all(isinstance(i, int) for i in offsets_hourly), "Hourly offsets not all integers"
    assert all((offsets_hourly[i] >= (720*i - 360)) and offsets_hourly[i] <= (720*i + 360) for i in range(0, len(offsets_hourly))), "Hourly offsets not in correct ranges"
    
    print(offsets_hourly[-1], total_time, forecast_steps) 
    assert all((offsets_hourly[-1] + step <= total_time for step in forecast_steps)), "Last hourly offset + forecast step exceeds total time"

def test_create_design_non_linear():
    """ tests the create_design_non_linear function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers

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

    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    # Create design matrix, daily
    non_linear_design = modelling_helpers.create_design_non_linear(daily_lags, daily_fourier_features, "D", daily_ts, "model_1")  
    assert list(non_linear_design.keys()) == ["model_1"], "Model name key incorrect"  
    X = non_linear_design["model_1"][0]
    y = non_linear_design["model_1"][1]
    dp = non_linear_design["model_1"][2]
    lags = non_linear_design["model_1"][3]


    # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
    # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
    assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Non-linear design matrix has incorrect number of rows"
    assert X.shape[1] == len(daily_lags) + 20 + 10, "Non-linear design matrix has incorrect number of columns"
    assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier), "Non-linear design matrix has incorrect column names"
    assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
    assert lags == daily_lags, "Lags returned incorrectly"


    # Create design matrix, hourly
    non_linear_design = modelling_helpers.create_design_non_linear(hourly_lags, hourly_fourier_features, "h", hourly_ts, "model_2")  
    assert list(non_linear_design.keys()) == ["model_2"], "Model name key incorrect"  
    X = non_linear_design["model_2"][0]
    y = non_linear_design["model_2"][1]
    dp = non_linear_design["model_2"][2]
    lags = non_linear_design["model_2"][3]


    # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
    assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "Non-linear design matrix has incorrect number of rows"
    assert X.shape[1] == len(hourly_lags) + 10 + 10, "Non-linear design matrix has incorrect number of columns"
    assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier), "Non-linear design matrix has incorrect column names"
    assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
    assert lags == hourly_lags, "Lags returned incorrectly"


def test_create_design_linear():
    """ tests the create_design_linear function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers

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

    # We will use order 3 so is at least interesting
    order = 3
    expected_order_terms = ["const", "trend", "trend_squared", "trend_cubed"]


    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    # Create design matrix, daily
    linear_design = modelling_helpers.create_design_linear(daily_lags, order, daily_fourier_features, "D", daily_ts, "model_1")  
    assert list(linear_design.keys()) == ["model_1"], "Model name key incorrect"  
    X = linear_design["model_1"][0]
    y = linear_design["model_1"][1]
    dp = linear_design["model_1"][2]
    lags = linear_design["model_1"][3]


    # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
    # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
    assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Linear design matrix has incorrect number of rows"
    assert X.shape[1] == len(daily_lags) + 20 + 10 + len(expected_order_terms), "Linear design matrix has incorrect number of columns"
    assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier + expected_order_terms), "Linear design matrix has incorrect column names"
    assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
    assert lags == daily_lags, "Lags returned incorrectly"


    # Create design matrix, hourly
    linear_design = modelling_helpers.create_design_linear(hourly_lags, order, hourly_fourier_features, "h", hourly_ts, "model_2")  
    assert list(linear_design.keys()) == ["model_2"], "Model name key incorrect"  
    X = linear_design["model_2"][0]
    y = linear_design["model_2"][1]
    dp = linear_design["model_2"][2]
    lags = linear_design["model_2"][3]


    # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
    assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "linear design matrix has incorrect number of rows"
    assert X.shape[1] == len(hourly_lags) + 10 + 10 + len(expected_order_terms), "linear design matrix has incorrect number of columns"
    assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier + expected_order_terms), "linear design matrix has incorrect column names"
    assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
    assert lags == hourly_lags, "Lags returned incorrectly"

def expected_xgbregressor_params() -> dict:
    """ returns the expected parameters for an XGBRegressor used in the tests

    Returns:
        dict: xgboost params
    """  
    from jfk_taxis import load_config

    config, PROJECT_ROOT = load_config()

    expected_params = {
        "n_estimators": config["xgboost_default"]["n_estimators"],
        "learning_rate":config["xgboost_default"]["learning_rate"],
        "max_depth":config["xgboost_default"]["max_depth"],
        "subsample":config["xgboost_default"]["subsample"],
        "colsample_bytree":config["xgboost_default"]["colsample_bytree"],
        "random_state":config["xgboost_setup"]["random_state"],
        "eval_metric":config["xgboost_setup"]["eval_metric"],
        "tree_method":config["xgboost_setup"]["tree_method"],
        "device":config["xgboost_setup"]["device"] 
    } 

    return expected_params

def default_xgb_model() -> XGBRegressor:
    """Create a default XGBRegressor model with predefined parameters.

    Returns:
        XGBRegressor: the default XGBRegressor
    """    
    params = expected_xgbregressor_params()

    model = XGBRegressor(**params)

    return model


def test_train_non_linear_models():
    """ tests the train_non_linear_models function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers
    from xgboost import XGBRegressor

    # We will reuse create_design_non_linear to actually make our non_linear_design dict

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

    # Expected parameters for the non linear model
    expected_params = expected_xgbregressor_params()

    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    # Create design matrix, daily
    non_linear_design_daily = modelling_helpers.create_design_non_linear(daily_lags, daily_fourier_features, "D", daily_ts, "model_1")

    # Train the model
    non_linear_models_daily = modelling_helpers.train_non_linear_models(non_linear_design_daily)

    # Checks
    assert list(non_linear_models_daily.keys()) == ["model_1"], "Model name key incorrect"
    
    # Unpack dict
    model = non_linear_models_daily["model_1"][0]
    dp = non_linear_models_daily["model_1"][1]
    hybrid = non_linear_models_daily["model_1"][2]
    lags = non_linear_models_daily["model_1"][3]

    # Get the model params
    model_params = model.get_params()

    # Again we have already checked that dps lags and fourier features are correct in test_preprocess
    assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
    assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
    for key in expected_params.keys():
        assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
    assert hybrid == None, "Hybrid should be None for non-hybrid model"  
    assert lags == daily_lags, "Lags returned incorrectly"

    # Repeat for hourly
    # Create design matrix, hourly
    non_linear_design_hourly = modelling_helpers.create_design_non_linear(hourly_lags, hourly_fourier_features, "h", hourly_ts, "model_1")

    # Train the model
    non_linear_models_hourly = modelling_helpers.train_non_linear_models(non_linear_design_hourly)

    # Checks
    assert list(non_linear_models_hourly.keys()) == ["model_1"], "Model name key incorrect"
    
    # Unpack dict
    model = non_linear_models_hourly["model_1"][0]
    dp = non_linear_models_hourly["model_1"][1]
    hybrid = non_linear_models_hourly["model_1"][2]
    lags = non_linear_models_hourly["model_1"][3]

    # Get the model params, note expected_params will only be a subset of this as we do not manually set all parameters
    model_params = model.get_params()

    # Again we have already checked that dps lags and fourier features are correct in test_preprocess
    assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
    assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
    for key in expected_params.keys():
        assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
    assert hybrid == None, "Hybrid should be None for non-hybrid model"  
    assert lags == hourly_lags, "Lags returned incorrectly"

def test_train_linear_models():
    """ tests the train_linear_models function in modelling_helpers.py
    """
    from jfk_taxis import modelling_helpers
    from sklearn.linear_model import LinearRegression

    # We will reuse create_design_linear to actually make our linear_design dict

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

    # We will use order 3 so is at least interesting
    order = 3
    expected_order_terms = ["const", "trend", "trend_squared", "trend_cubed"]

    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    # Create design matrix, daily
    linear_design_daily = modelling_helpers.create_design_linear(daily_lags, order, daily_fourier_features, "D", daily_ts, "model_1")

    # Train the model
    linear_models_daily = modelling_helpers.train_linear_models(linear_design_daily)

    # Checks
    assert list(linear_models_daily.keys()) == ["model_1"], "Model name key incorrect"
    
    # Unpack dict
    model = linear_models_daily["model_1"][0]
    dp = linear_models_daily["model_1"][1]
    hybrid = linear_models_daily["model_1"][2]
    lags = linear_models_daily["model_1"][3]

    # Again we have already checked that dps lags and fourier features are correct in test_preprocess
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
    assert hybrid == None, "Hybrid should be None for non-hybrid model"  
    assert lags == daily_lags, "Lags returned incorrectly"

    # Repeat for hourly
    # Create design matrix, hourly
    linear_design_hourly = modelling_helpers.create_design_linear(hourly_lags, order, hourly_fourier_features, "h", hourly_ts, "model_1")

    # Train the model
    linear_models_hourly = modelling_helpers.train_linear_models(linear_design_hourly)

    # Checks
    assert list(linear_models_hourly.keys()) == ["model_1"], "Model name key incorrect"

    # Unpack dict
    model = linear_models_hourly["model_1"][0]
    dp = linear_models_hourly["model_1"][1]
    hybrid = linear_models_hourly["model_1"][2]
    lags = linear_models_hourly["model_1"][3]

    # Again we have already checked that dps lags and fourier features are correct in test_preprocess
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
    assert hybrid == None, "Hybrid should be None for non-hybrid model"  
    assert lags == hourly_lags, "Lags returned incorrectly"

def test_train_hybrid_models():
    """ tests the train_hybrid_models function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers
    from sklearn.linear_model import LinearRegression
    import copy

    # We will reuse create_design_linear to actually make our linear_design dict

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

    # We will use order 3 so is at least interesting
    order = 3
    expected_order_terms = ["const", "trend", "trend_squared", "trend_cubed"]

    # XGBRegressor to use
    xgbregressor = default_xgb_model()

    # Expected params for this model
    expected_params = expected_xgbregressor_params()

    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    # Create design matrix, daily
    linear_design_daily = modelling_helpers.create_design_linear(daily_lags, order, daily_fourier_features, "D", daily_ts, "model_1")

    # Train the model
    hybrid = copy.deepcopy(xgbregressor) # so we don't get errors from fitting same instance twice
    linear_models_daily = modelling_helpers.train_hybrid_models(linear_design_daily, hybrid) 

    # Checks
    assert list(linear_models_daily.keys()) == ["model_1"], "Model name key incorrect"
    
    # Unpack dict
    model = linear_models_daily["model_1"][0]
    dp = linear_models_daily["model_1"][1]
    hybrid = linear_models_daily["model_1"][2]
    lags = linear_models_daily["model_1"][3]

    # Get the hybrid params, note expected_params will only be a subset of this as we do not manually set all parameters
    hybrid_params = hybrid.get_params()

    # Again we have already checked that dps lags and fourier features are correct in test_preprocess
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
    assert set(expected_params.keys()) <= set(hybrid_params.keys()), "Expected params not a subset of model params"
    for key in expected_params.keys():
        assert hybrid_params[key] == expected_params[key], "Model params and expected params differ in values"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
    assert isinstance(hybrid, XGBRegressor), "Hybrid model is not an XGBRegressor instance"
    assert lags == daily_lags, "Lags returned incorrectly"

    # Repeat for hourly
    # Create design matrix, hourly
    linear_design_hourly = modelling_helpers.create_design_linear(hourly_lags, order, hourly_fourier_features, "h", hourly_ts, "model_1")

    # Train the model
    hybrid = copy.deepcopy(xgbregressor)
    linear_models_hourly = modelling_helpers.train_hybrid_models(linear_design_hourly, hybrid)

    # Checks
    assert list(linear_models_hourly.keys()) == ["model_1"], "Model name key incorrect"

    # Unpack dict
    model = linear_models_hourly["model_1"][0]
    dp = linear_models_hourly["model_1"][1]
    hybrid = linear_models_hourly["model_1"][2]
    lags = linear_models_hourly["model_1"][3]

    # Get the hybrid params, note expected_params will only be a subset of this as we do not manually set all parameters
    hybrid_params = hybrid.get_params()

    # Again we have already checked that dps lags and fourier features are correct in test_preprocess
    assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
    assert set(expected_params.keys()) <= set(hybrid_params.keys()), "Expected params not a subset of model params"
    for key in expected_params.keys():
        assert hybrid_params[key] == expected_params[key], "Model params and expected params differ in values"
    assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
    assert isinstance(hybrid, XGBRegressor), "Hybrid model is not an XGBRegressor instance"
    assert lags == hourly_lags, "Lags returned incorrectly"

def test_create_train_non_linear():
    """ test the create_train_non_linear function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers

    # Essentially this function just combines the create_design_non_linear and train_non_linear_models so we will just combine the tests essentially
    # and check we get what we expect

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


    # Expected params for this model
    expected_params = expected_xgbregressor_params()

    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    # To make sure we are actually creating two models (that will be identical)
    daily_names = ["model_1", "model_2"]
    hourly_names = ["model_2", "model_3"]

    # Create and train model, daily
    non_linear_design, non_linear_models_daily = modelling_helpers.create_train_non_linear(daily_names, daily_lags, daily_fourier_features, "D", daily_ts)

    # Checks (these are just the create design and train model checks combined)
    for name in daily_names:
        assert set(non_linear_design.keys()) == set(daily_names), "Model name key incorrect"
        X = non_linear_design[name][0]
        y = non_linear_design[name][1]
        dp = non_linear_design[name][2]
        lags = non_linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Non-linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10, "Non-linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier), "Non-linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"
    
        assert set(list(non_linear_models_daily.keys())) == set(daily_names), "Model name key incorrect"
        
        # Unpack dict
        model = non_linear_models_daily[name][0]
        dp = non_linear_models_daily[name][1]
        hybrid = non_linear_models_daily[name][2]
        lags = non_linear_models_daily[name][3]

        # Get the model params
        model_params = model.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
        assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == daily_lags, "Lags returned incorrectly"

    # Repeat for hourly
    non_linear_design, non_linear_models_hourly = modelling_helpers.create_train_non_linear(hourly_names, hourly_lags, hourly_fourier_features, "h", hourly_ts)

    for name in hourly_names:
        # Upack non linear design
        assert set(list(non_linear_design.keys())) == set(hourly_names), "Model name key incorrect"  
        X = non_linear_design[name][0]
        y = non_linear_design[name][1]
        dp = non_linear_design[name][2]
        lags = non_linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "Non-linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10, "Non-linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier), "Non-linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Check trained model
        assert set(non_linear_models_hourly.keys()) == set(hourly_names), "Model name key incorrect"
        
        # Unpack dict
        model = non_linear_models_hourly[name][0]
        dp = non_linear_models_hourly[name][1]
        hybrid = non_linear_models_hourly[name][2]
        lags = non_linear_models_hourly[name][3]

        # Get the model params, note expected_params will only be a subset of this as we do not manually set all parameters
        model_params = model.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
        assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == hourly_lags, "Lags returned incorrectly"

def test_create_train_linear():
    """ test the create_train_linear function in modelling_helpers.py
    """
    from jfk_taxis import modelling_helpers
    from sklearn.linear_model import LinearRegression

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


    daily_ts = create_ts("D")
    hourly_ts = create_ts("h")

    daily_names = ["model_1", "model_2"]
    hourly_names = ["model_2", "model_3"]

    # Daily case
    linear_design, linear_models_daily = modelling_helpers.create_train_linear(daily_names, order_list, daily_lags, daily_fourier_features, "D", daily_ts)

    for name, expected_order_terms in zip(daily_names, [expected_3_order_terms, expected_2_order_terms]):
        # Checks on design
        assert set(list(linear_design.keys())) == set(daily_names), "Model name key incorrect"
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10 + len(expected_order_terms), "Linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier + expected_order_terms), "Linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"

        # Model checks
        assert set(list(linear_models_daily.keys())) == set(daily_names), "Model name key incorrect"

        # Unpack dict
        model = linear_models_daily[name][0]
        dp = linear_models_daily[name][1]
        hybrid = linear_models_daily[name][2]
        lags = linear_models_daily[name][3]

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == daily_lags, "Lags returned incorrectly"

    # Hourly case
    linear_design, linear_models_hourly = modelling_helpers.create_train_linear(hourly_names, order_list, hourly_lags, hourly_fourier_features, "h", hourly_ts)

    for name, expected_order_terms in zip(hourly_names, [expected_3_order_terms, expected_2_order_terms]):
        # Design checks 
        assert set(list(linear_design.keys())) == set(hourly_names), "Model name key incorrect"
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]

        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10 + len(expected_order_terms), "linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier + expected_order_terms), "linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Model checks 
        assert set(list(linear_models_hourly.keys())) == {"model_2", "model_3"}, "Model name key incorrect"

        # Unpack dict
        model = linear_models_hourly[name][0]
        dp = linear_models_hourly[name][1]
        hybrid = linear_models_hourly[name][2]
        lags = linear_models_hourly[name][3]

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == hourly_lags, "Lags returned incorrectly"

def test_create_train_hybrid():
    """ tests the create_train_hybrid function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers
    from sklearn.linear_model import LinearRegression
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

    daily_names = ["model_1", "model_2"]
    hourly_names = ["model_2", "model_3"]

    # Create and train model, daily
    hybrid = copy.deepcopy(xgbregressor) # so we don't get errors from fitting same instance twice
    linear_design, linear_models_daily = modelling_helpers.create_train_hybrid(daily_names, hybrid, order_list, daily_lags, daily_fourier_features, "D", daily_ts)

    # Checks
    for name, expected_order_terms in zip(daily_names, [expected_3_order_terms, expected_2_order_terms]):
        assert set(list(linear_design.keys())) == set(daily_names), "Model name key incorrect"  
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10 + len(expected_order_terms), "Linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier + expected_order_terms), "Linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"

        assert set(list(linear_models_daily.keys())) == set(daily_names), "Model name key incorrect"
        
        # Unpack dict
        model = linear_models_daily[name][0]
        dp = linear_models_daily[name][1]
        hybrid = linear_models_daily[name][2]
        lags = linear_models_daily[name][3]

        # Get the hybrid params, note expected_params will only be a subset of this as we do not manually set all parameters
        hybrid_params = hybrid.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(expected_params.keys()) <= set(hybrid_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert hybrid_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert isinstance(hybrid, XGBRegressor), "Hybrid model is not an XGBRegressor instance"
        assert lags == daily_lags, "Lags returned incorrectly"
    
    # Repeat for hourly
    hybrid = copy.deepcopy(xgbregressor)
    linear_design, linear_models_hourly = modelling_helpers.create_train_hybrid(hourly_names, hybrid, order_list, hourly_lags, hourly_fourier_features, "h", hourly_ts)

    for name, expected_order_terms in zip(hourly_names, [expected_3_order_terms, expected_2_order_terms]):
        # Design checks 
        assert set(list(linear_design.keys())) == set(hourly_names), "Model name key incorrect"  
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10 + len(expected_order_terms), "linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier + expected_order_terms), "linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Model checks
        assert set(list(linear_models_hourly.keys())) == set(hourly_names), "Model name key incorrect"

        # Unpack dict
        model = linear_models_hourly[name][0]
        dp = linear_models_hourly[name][1]
        hybrid = linear_models_hourly[name][2]
        lags = linear_models_hourly[name][3]

        # Get the hybrid params, note expected_params will only be a subset of this as we do not manually set all parameters
        hybrid_params = hybrid.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(expected_params.keys()) <= set(hybrid_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert hybrid_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert isinstance(hybrid, XGBRegressor), "Hybrid model is not an XGBRegressor instance"
        assert lags == hourly_lags, "Lags returned incorrectly"

def test_create_train_save_models():
    """ test the create_train_save_models function in modelling_helpers.py
    """    
    from jfk_taxis import modelling_helpers
    from jfk_taxis import load_models, load_design
    import copy
    from sklearn.linear_model import LinearRegression

    # This function just combines all the previous functions so we will essentially just run the same checks again
    # We will need to do two sets one for linear and one for hybrid
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

    # Load the models and designs back in and perform checks

    # Daily linear
    linear_design, non_linear_design = load_design(daily_linear_sig)
    linear_models_daily, non_linear_models_daily = load_models(daily_linear_sig)

    # Check linear models and design
    for name, expected_order_terms in zip(daily_names_linear, [expected_3_order_terms, expected_2_order_terms]):
        # Checks on design
        assert set(list(linear_design.keys())) == set(daily_names_linear), "Model name key incorrect"
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10 + len(expected_order_terms), "Linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier + expected_order_terms), "Linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"

        # Model checks
        assert set(list(linear_models_daily.keys())) == set(daily_names_linear), "Model name key incorrect"

        # Unpack dict
        model = linear_models_daily[name][0]
        dp = linear_models_daily[name][1]
        hybrid = linear_models_daily[name][2]
        lags = linear_models_daily[name][3]

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == daily_lags, "Lags returned incorrectly"

    # Check non-linear models and design 
    for name in daily_names_non_linear:
        assert set(non_linear_design.keys()) == set(daily_names_non_linear), "Model name key incorrect"
        X = non_linear_design[name][0]
        y = non_linear_design[name][1]
        dp = non_linear_design[name][2]
        lags = non_linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Non-linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10, "Non-linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier), "Non-linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"
    
        assert set(list(non_linear_models_daily.keys())) == set(daily_names_non_linear), "Model name key incorrect"

        # Unpack dict
        model = non_linear_models_daily[name][0]
        dp = non_linear_models_daily[name][1]
        hybrid = non_linear_models_daily[name][2]
        lags = non_linear_models_daily[name][3]

        # Get the model params
        model_params = model.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
        assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == daily_lags, "Lags returned incorrectly"

    # Daily hybrid (we still with the naming convention linear_design even though it is hybrid here, the model is a linear regression boosted on residuals)
    linear_design, non_linear_design = load_design(daily_hybrid_sig)
    linear_models_daily, non_linear_models_daily = load_models(daily_hybrid_sig)

    # Checks hybrid model
    for name, expected_order_terms in zip(daily_names_hybrid, [expected_3_order_terms, expected_2_order_terms]):
        assert set(list(linear_design.keys())) == set(daily_names_hybrid), "Model name key incorrect"  
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10 + len(expected_order_terms), "Linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier + expected_order_terms), "Linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"

        assert set(list(linear_models_daily.keys())) == set(daily_names_hybrid), "Model name key incorrect"
        
        # Unpack dict
        model = linear_models_daily[name][0]
        dp = linear_models_daily[name][1]
        hybrid = linear_models_daily[name][2]
        lags = linear_models_daily[name][3]

        # Get the hybrid params, note expected_params will only be a subset of this as we do not manually set all parameters
        hybrid_params = hybrid.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(expected_params.keys()) <= set(hybrid_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert hybrid_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert isinstance(hybrid, XGBRegressor), "Hybrid model is not an XGBRegressor instance"
        assert lags == daily_lags, "Lags returned incorrectly"

    # Checks non linear model
    for name in daily_names_non_linear:
        assert set(non_linear_design.keys()) == set(daily_names_non_linear), "Model name key incorrect"
        X = non_linear_design[name][0]
        y = non_linear_design[name][1]
        dp = non_linear_design[name][2]
        lags = non_linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        # A lot of these checks are similar to those in test_preprocess where there are more explanations to the constants
        assert X.shape[0] == daily_ts.shape[0] - daily_lags[-1], "Non-linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(daily_lags) + 20 + 10, "Non-linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_daily_lags + expected_daily_fourier), "Non-linear design matrix has incorrect column names"
        assert y.shape[0] == daily_ts.shape[0] - daily_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
        assert lags == daily_lags, "Lags returned incorrectly"
    
        assert set(list(non_linear_models_daily.keys())) == set(daily_names_non_linear), "Model name key incorrect"

        # Unpack dict
        model = non_linear_models_daily[name][0]
        dp = non_linear_models_daily[name][1]
        hybrid = non_linear_models_daily[name][2]
        lags = non_linear_models_daily[name][3]

        # Get the model params
        model_params = model.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
        assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == daily_lags, "Lags returned incorrectly"

    # Repeat for hourly linear
    linear_design, non_linear_design = load_design(hourly_linear_sig)
    linear_models_hourly, non_linear_models_hourly = load_models(hourly_linear_sig)

    # Check linear models and design
    for name, expected_order_terms in zip(hourly_names_linear, [expected_3_order_terms, expected_2_order_terms]):
        # Design checks 
        assert set(list(linear_design.keys())) == set(hourly_names_linear), "Model name key incorrect"
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]

        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10 + len(expected_order_terms), "linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier + expected_order_terms), "linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Model checks 
        assert set(list(linear_models_hourly.keys())) == {"model_2", "model_3"}, "Model name key incorrect"

        # Unpack dict
        model = linear_models_hourly[name][0]
        dp = linear_models_hourly[name][1]
        hybrid = linear_models_hourly[name][2]
        lags = linear_models_hourly[name][3]

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == hourly_lags, "Lags returned incorrectly"

    # Check non-linear models and design
    for name in hourly_names_non_linear:
        # Unpack non linear design
        assert set(list(non_linear_design.keys())) == set(hourly_names_non_linear), "Model name key incorrect"  
        X = non_linear_design[name][0]
        y = non_linear_design[name][1]
        dp = non_linear_design[name][2]
        lags = non_linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "Non-linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10, "Non-linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier), "Non-linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Check trained model
        assert set(non_linear_models_hourly.keys()) == set(hourly_names_non_linear), "Model name key incorrect"
        
        # Unpack dict
        model = non_linear_models_hourly[name][0]
        dp = non_linear_models_hourly[name][1]
        hybrid = non_linear_models_hourly[name][2]
        lags = non_linear_models_hourly[name][3]

        # Get the model params, note expected_params will only be a subset of this as we do not manually set all parameters
        model_params = model.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
        assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == hourly_lags, "Lags returned incorrectly"

    # Repeat for hourly hybrid
    linear_design, non_linear_design = load_design(hourly_hybrid_sig)
    linear_models_hourly, non_linear_models_hourly = load_models(hourly_hybrid_sig)

    # Checks hybrid model and design
    for name, expected_order_terms in zip(hourly_names_hybrid, [expected_3_order_terms, expected_2_order_terms]):
        # Design checks 
        assert set(list(linear_design.keys())) == set(hourly_names_hybrid), "Model name key incorrect"  
        X = linear_design[name][0]
        y = linear_design[name][1]
        dp = linear_design[name][2]
        lags = linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10 + len(expected_order_terms), "linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier + expected_order_terms), "linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Model checks
        assert set(list(linear_models_hourly.keys())) == set(hourly_names_hybrid), "Model name key incorrect"

        # Unpack dict
        model = linear_models_hourly[name][0]
        dp = linear_models_hourly[name][1]
        hybrid = linear_models_hourly[name][2]
        lags = linear_models_hourly[name][3]

        # Get the hybrid params, note expected_params will only be a subset of this as we do not manually set all parameters
        hybrid_params = hybrid.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, LinearRegression), "Model is not a LinearRegression instance"
        assert set(expected_params.keys()) <= set(hybrid_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert hybrid_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), "Deterministic Process has incorrect columns"
        assert isinstance(hybrid, XGBRegressor), "Hybrid model is not an XGBRegressor instance"
        assert lags == hourly_lags, "Lags returned incorrectly"

    # Checks non linear model and designs 
    for name in hourly_names_non_linear:
        # Upack non linear design
        assert set(list(non_linear_design.keys())) == set(hourly_names_non_linear), "Model name key incorrect"  
        X = non_linear_design[name][0]
        y = non_linear_design[name][1]
        dp = non_linear_design[name][2]
        lags = non_linear_design[name][3]


        # Checks (note we have already checked preprocess implements lags and fourier features correctly in test_preprocess)
        assert X.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "Non-linear design matrix has incorrect number of rows"
        assert X.shape[1] == len(hourly_lags) + 10 + 10, "Non-linear design matrix has incorrect number of columns"
        assert set(X.columns.tolist()) == set(expected_hourly_lags + expected_hourly_fourier), "Non-linear design matrix has incorrect column names"
        assert y.shape[0] == hourly_ts.shape[0] - hourly_lags[-1], "y has incorrect number of rows"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
        assert lags == hourly_lags, "Lags returned incorrectly"

        # Check trained model
        assert set(non_linear_models_hourly.keys()) == set(hourly_names_non_linear), "Model name key incorrect"

        # Unpack dict
        model = non_linear_models_hourly[name][0]
        dp = non_linear_models_hourly[name][1]
        hybrid = non_linear_models_hourly[name][2]
        lags = non_linear_models_hourly[name][3]

        # Get the model params, note expected_params will only be a subset of this as we do not manually set all parameters
        model_params = model.get_params()

        # Again we have already checked that dps lags and fourier features are correct in test_preprocess
        assert isinstance(model, XGBRegressor), "Model is not a XGBRegressor instance"
        assert set(expected_params.keys()) <= set(model_params.keys()), "Expected params not a subset of model params"
        for key in expected_params.keys():
            assert model_params[key] == expected_params[key], "Model params and expected params differ in values"
        assert set(dp.out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), "Deterministic Process has incorrect columns"
        assert hybrid == None, "Hybrid should be None for non-hybrid model"  
        assert lags == hourly_lags, "Lags returned incorrectly"

     
