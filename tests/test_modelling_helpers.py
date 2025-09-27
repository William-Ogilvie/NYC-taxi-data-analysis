"""
test_modelling_helpers.py
==========================

Unit tests for modelling_helpers.py
"""
import pytest
import pandas as pd

def create_ts(time_step: str) -> pd.Series:
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


