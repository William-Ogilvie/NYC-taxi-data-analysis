"""
test_forecast_helpers.py
=========================

Unit tests for forecast helpers.py. Note to_numpy, fit_linear, fit_non_linear are just wrappers around sklearn functions so don't need their own unit tests.
truncate_lags is a simple function that just truncates a list so doesn't need its own unit test.

We will also not test forecast_dicts or run_forecasts, this is because these functions essentially just call the other functions and plot the output, so we have checked their plots themselves.
In a similar vein we will not test any of the other functions that only plot things like create_avg_mae_barplot, however we will test the componenets they rely on like create_avg_mae_df.
"""

import pytest
from statsmodels.tsa.deterministic import DeterministicProcess
import pandas as pd

def test_drop_time_zone():
    """ test for drop_time_zone function in forecast_helpers.py, note this function only works if the timezone is UTC (which it will be as we do all computations in UTC).
    """

    import pandas as pd
    from jfk_taxis import forecast_helpers
    from pandas import Timestamp

    series = pd.Series(data = [1, 2, 3], index = pd.to_datetime([Timestamp("2023-01-01 00:00:00+00:00", tz="UTC"),
                                      Timestamp("2023-05-25 01:00:00+00:00", tz ="UTC"),
                                      Timestamp("2011-12-31 23:00:00+00:00", tz="UTC")]))

    result = forecast_helpers.drop_time_zone(series)

    # Checks
    assert all(result == pd.Series(data = [1, 2, 3], index = [Timestamp("2023-01-01 00:00:00"),
                                      Timestamp("2023-05-25 01:00:00"),
                                      Timestamp("2011-12-31 23:00:00")])), "Values or index incorrect"


def compute_fourier_feature(k: int, t: int, m: int, sin_or_cos: str) -> float:
    """ Compute a single fourier feature value

    Args:
        k (int): harmonic
        t (int): time index
        m (int): period
        sin_or_cos (str): "sin" or "cos"

    Returns:
        float: the fourier feature value
    """    
    import numpy as np

    if sin_or_cos == "sin":
        return np.sin((2 * np.pi * k * t) / m)
    elif sin_or_cos == "cos":
        return np.cos((2 * np.pi * k * t) / m)
    else:
        raise ValueError("sin_or_cos must be 'sin' or 'cos'")
    
def compute_weekly_fourier_features(date_time: pd.Timestamp, harmonic: int, sin_or_cos: str, time_step: str) -> float:
    """ Compute the weekly fourier feature for a given date_time and harmonic

    Args:
        date_time (pd.Timestamp): the date time
        harmonic (int): the harmonic
        sin_or_cos (str): "sin" or "cos"
        time_step (str): "D" for daily data, "h" for hourly data

    Returns:
        float: the weekly fourier feature value
    """    

    # First work out what day of the week it is, this will be t
    t = date_time.dayofweek # Monday = 0, Sunday = 6
    if time_step == "D":
        # For just daily series there are 7 days in a week so m = 7
        m = 7
    elif time_step == "h":
        # Now the t for hourly series will be the day of the week * 24 + the hour of the day
        t = t * 24 + date_time.hour # 0 to 167
        # For hourly series there are 24 hours in a day and 7 days in a week so m = 24 * 7
        m = 24 * 7
    else:
        raise ValueError("time_step must be 'D' or 'h'")
    # Now k will be the harmonic
    k = harmonic

    return compute_fourier_feature(k, t, m, sin_or_cos)

def compute_yearly_fourier_features(date_time: pd.Timestamp, harmonic: int, sin_or_cos: str) -> float:
    """ Compute the yearly fourier feature for a given date_time and harmonic

    Args:
        date_time (pd.Timestamp): the date time
        harmonic (int): the harmonic
        sin_or_cos (str): "sin" or "cos"

    Returns:
        float: the yearly fourier feature value
    """    

    # First work out what day of the year it is, this will be t
    t = date_time.day_of_year # 1 to 365 (or 366)
    # Now for freq=YE-DEC 0 is Jan 1st, so we need to subtract 1 from t and take % 365
    t = (t - 1) % 365
    # There are 365 days in a year so m = 365
    m = 365
    # Now k will be the harmonic
    k = harmonic
    return compute_fourier_feature(k, t, m, sin_or_cos)

def compute_daily_fourier_features(date_time: pd.Timestamp, harmonic: int, sin_or_cos: str) -> float:
    """ Compute the daily fourier feature for a given date_time and harmonic

    Args:
        date_time (pd.Timestamp): the date time
        harmonic (int): the harmonic
        sin_or_cos (str): "sin" or "cos"

    Returns:
        float: the daily fourier feature value
    """    

    # Work out what hour of the day it is, this will be t
    t = date_time.hour # 0 to 23
    # There are 24 hours in a day so m = 24
    m = 24
    # Now k will be the harmonic
    k = harmonic
    return compute_fourier_feature(k, t, m, sin_or_cos)

def check_lags_and_fourier_preprocess(dp: DeterministicProcess, X: pd.DataFrame, lags: list[int], y: pd.Series, time_step: str) -> bool:
    """ checks that the lags and fourier features implemented by the preprocess have been done correctly

    Args:
        dp (DeterministicProcess): deterministic process in question
        X (pd.DataFrame): design matrix
        lags (list[int]): list of lags to check
        y (pd.Series): original time series
        time_step (str): "D" for daily data, "h" for hourly data

    Returns:
        bool: True if the lags are correctly applied, False otherwise
    """    

    # Check each lag one by one
    for lag in lags:

        # First check the design matrix
        for i in range(0, X.shape[0]):
            # Check the lag has been implemented correctly
            # Value in design matrix
            design_val =  X.iloc[i][f"y_lag_{lag}"]

            # The value of this index
            index_val = X.index[i]

            # Convert to datetime with UTC
            index_val = pd.to_datetime(index_val, utc = True)

            # Find the postion in y
            pos = y.index.get_loc(index_val)

            # Find the value in y "lag" time steps backwards
            real_lag = y.iloc[pos - lag]

            # Check that these two values are equal
            if real_lag == design_val:
                continue
            else:
                print("Found mismatch in design matrix lags and true lags")
                print(i)
                print(design_val)
                print(real_lag)
                print(y.iloc[:-5])
                return False
    
    # Go out 30 steps into the deterministic process and check 
    # The formula for the fourier features is:
    # sin(2 * pi * k * t / m) and cos(2 * pi * k * t / m)
    # where k is the harmonic, t is the time index, and m is the period
    step = 30
    X_out = dp.out_of_sample(steps=step)
    for i in range(0, X_out.shape[0]):
        for col, value in X_out.iloc[i].items(): 
            # Check if this is a fourier feature (general structure is sin(x,freq=Y-Z) or cos(x,freq=Y-Z))
            if ("sin" in col) or ("cos" in col):
                sin_or_cos = col.split("(")[0] # "sin" or "cos"
                order = col.split("(")[1].split(",")[0] # the harmonic
                freq = col.split("freq=")[1].split("-")[0].split(")")[0] # the frequency (YE, W, D)

                if freq == "W":
                    expected_value = compute_weekly_fourier_features(X_out.index[i], int(order), sin_or_cos, time_step)
                elif freq == "YE":
                    expected_value = compute_yearly_fourier_features(X_out.index[i], int(order), sin_or_cos)
                elif freq == "D":
                    expected_value = compute_daily_fourier_features(X_out.index[i], int(order), sin_or_cos)
                else:
                    raise ValueError(f"Unknown frequency {freq} in column {col}")
                
                # Compare this to our given value with some tolerance due to floating point errors
                if abs(expected_value - value) > 1e-1:
                    print("Found mismatch in fourier feature computation")
                    print(X_out.index[i])
                    print(col)
                    print(value)
                    print(expected_value)
                    return False 

    return True

    

def test_preprocess():
    """ test for preprocess function in forecast_helpers.py
    """
    import pandas as pd
    from jfk_taxis import forecast_helpers
    


    # So that we can use the fourier features we will create 3 years worth of daily and hourly time series data
    series_daily_full = pd.Series(data = range(1, 365*3 + 1), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3, freq = "D"))
    series_hourly_full = pd.Series(data = range(1, 365*3*24 + 1), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3*24, freq = "h"))

    # To make it interesting we will remove the 2nd to last row from both the daily and hourly series 
    series_daily = series_daily_full.drop(index = series_daily_full.index[-2])
    series_hourly = series_hourly_full.drop(index = series_hourly_full.index[-2])

    # Now as we want to use the full series later when we check lags but the future value at that index will actually be 0 (because it gets filled in as 0), so we need to set it to be 0 now so we can check the lags properly
    series_daily_full.iloc[-2] = 0
    series_hourly_full.iloc[-2] = 0

    # Lags
    daily_lags = [1, 2, 7, 23, 364]
    hourly_lags = [1, 2, 24, 48, 24*365+12]

    # Expected lags
    expected_daily_lags = [f"y_lag_{lag}" for lag in daily_lags]
    expected_hourly_lags = [f"y_lag_{lag}" for lag in hourly_lags]

    # Fourier features
    daily_fourier = ["YE", "W"]
    hourly_fourier = ["D", "W"]

    # Expected daily fourier features
    expected_daily_fourier_features = ['sin(1,freq=YE-DEC)', 'cos(1,freq=YE-DEC)', 'sin(2,freq=YE-DEC)',
       'cos(2,freq=YE-DEC)', 'sin(3,freq=YE-DEC)', 'cos(3,freq=YE-DEC)',
       'sin(4,freq=YE-DEC)', 'cos(4,freq=YE-DEC)', 'sin(5,freq=YE-DEC)',
       'cos(5,freq=YE-DEC)', 'sin(6,freq=YE-DEC)', 'cos(6,freq=YE-DEC)',
       'sin(7,freq=YE-DEC)', 'cos(7,freq=YE-DEC)', 'sin(8,freq=YE-DEC)',
       'cos(8,freq=YE-DEC)', 'sin(9,freq=YE-DEC)', 'cos(9,freq=YE-DEC)',
       'sin(10,freq=YE-DEC)', 'cos(10,freq=YE-DEC)', 'sin(1,freq=W-SUN)',
       'cos(1,freq=W-SUN)', 'sin(2,freq=W-SUN)', 'cos(2,freq=W-SUN)',
       'sin(3,freq=W-SUN)', 'cos(3,freq=W-SUN)', 'sin(4,freq=W-SUN)',
       'cos(4,freq=W-SUN)', 'sin(5,freq=W-SUN)', 'cos(5,freq=W-SUN)']
    
    # Expected hourly fourier features
    expected_hourly_fourier_features = ['sin(1,freq=D)', 'cos(1,freq=D)', 'sin(2,freq=D)',
       'cos(2,freq=D)', 'sin(3,freq=D)', 'cos(3,freq=D)', 'sin(4,freq=D)',
       'cos(4,freq=D)', 'sin(5,freq=D)', 'cos(5,freq=D)', 'sin(1,freq=W-SUN)',
       'cos(1,freq=W-SUN)', 'sin(2,freq=W-SUN)', 'cos(2,freq=W-SUN)',
       'sin(3,freq=W-SUN)', 'cos(3,freq=W-SUN)', 'sin(4,freq=W-SUN)',
       'cos(4,freq=W-SUN)', 'sin(5,freq=W-SUN)', 'cos(5,freq=W-SUN)']
    

    # Preprocess, we will do order 0 and order 2 as well as no constant and constant
    daily_order_0_no_const = forecast_helpers.preprocess(daily_lags, False, 0, daily_fourier, "D", series_daily)
    daily_order_0_const = forecast_helpers.preprocess(daily_lags, True, 0, daily_fourier, "D", series_daily)
    daily_order_2_no_const = forecast_helpers.preprocess(daily_lags, False, 2, daily_fourier, "D", series_daily)
    daily_order_2_const = forecast_helpers.preprocess(daily_lags, True, 2, daily_fourier, "D", series_daily)
    hourly_order_0_no_const = forecast_helpers.preprocess(hourly_lags, False, 0, hourly_fourier, "h", series_hourly)
    hourly_order_0_const = forecast_helpers.preprocess(hourly_lags, True, 0, hourly_fourier, "h", series_hourly)
    hourly_order_2_no_const = forecast_helpers.preprocess(hourly_lags, False, 2, hourly_fourier, "h", series_hourly)
    hourly_order_2_const = forecast_helpers.preprocess(hourly_lags, True, 2, hourly_fourier, "h", series_hourly)

    # Checks note we have the tuple (X, y, dp, lags)
    # Unpack the tuples
    X_daily_order_0_no_const, y_daily_order_0_no_const, dp_daily_order_0_no_const, lags_daily_order_0_no_const = daily_order_0_no_const
    X_daily_order_0_const, y_daily_order_0_const, dp_daily_order_0_const, lags_daily_order_0_const = daily_order_0_const
    X_daily_order_2_no_const, y_daily_order_2_no_const, dp_daily_order_2_no_const, lags_daily_order_2_no_const = daily_order_2_no_const
    X_daily_order_2_const, y_daily_order_2_const, dp_daily_order_2_const, lags_daily_order_2_const = daily_order_2_const
    X_hourly_order_0_no_const, y_hourly_order_0_no_const, dp_hourly_order_0_no_const, lags_hourly_order_0_no_const = hourly_order_0_no_const
    X_hourly_order_0_const, y_hourly_order_0_const, dp_hourly_order_0_const, lags_hourly_order_0_const = hourly_order_0_const
    X_hourly_order_2_no_const, y_hourly_order_2_no_const, dp_hourly_order_2_no_const, lags_hourly_order_2_no_const = hourly_order_2_no_const
    X_hourly_order_2_const, y_hourly_order_2_const, dp_hourly_order_2_const, lags_hourly_order_2_const = hourly_order_2_const   

    # Basic size checks 
    assert X_daily_order_0_no_const.shape[0] == X_daily_order_0_const.shape[0] == X_daily_order_2_no_const.shape[0] == X_daily_order_2_const.shape[0] == 1095 - daily_lags[-1] # as we have to drop the first daily_lags[-1] rows due to creating the lags
    assert X_hourly_order_0_no_const.shape[0] == X_hourly_order_0_const.shape[0] == X_hourly_order_2_no_const.shape[0] == X_hourly_order_2_const.shape[0] == 26280 - hourly_lags[-1] 
    assert X_daily_order_0_no_const.shape[1] == 20 + 10 + len(daily_lags) # 20 yearly fourier, 10 weekly, plus lags
    assert X_daily_order_0_const.shape[1] == 20 + 10 + len(daily_lags) + 1 # 20 yearly fourier, 10 weekly, plus lags plus constant
    assert X_daily_order_2_no_const.shape[1] == 20 + 10 + len(daily_lags) + 2 # 20 yearly fourier, 10 weekly, plus trend and trend^2
    assert X_daily_order_2_const.shape[1] == 20 + 10 + len(daily_lags) + 2 + 1 # 20 yearly fourier, 10 weekly, plus trend and trend^2 plus constant
    assert X_hourly_order_0_no_const.shape[1] == 10 + 10 + len(hourly_lags) # 10 daily fourier, 10 weekly, plus lags
    assert X_hourly_order_0_const.shape[1] == 10 + 10 + len(hourly_lags) + 1 # 10 daily fourier, 10 weekly, plus lags plus constant
    assert X_hourly_order_2_no_const.shape[1] == 10 + 10 + len(hourly_lags) + 2 # 10 daily fourier, 10 weekly, plus trend and trend^2
    assert X_hourly_order_2_const.shape[1] == 10 + 10 + len(hourly_lags) + 2 + 1 # 10 daily fourier, 10 weekly, plus trend and trend^2 plus constant

    # Check the feature column names
    temp_expected = set(expected_daily_fourier_features + expected_daily_lags) # we use a set so order doesn't matter
    assert set(X_daily_order_0_no_const.columns) == temp_expected
    temp_expected = set(expected_daily_fourier_features + expected_daily_lags + ["const"])
    assert set(X_daily_order_0_const.columns) == temp_expected
    temp_expected = set(expected_daily_fourier_features + expected_daily_lags + ["trend", "trend_squared"])
    assert set(X_daily_order_2_no_const.columns) == temp_expected
    temp_expected = set(expected_daily_fourier_features + expected_daily_lags + ["const", "trend", "trend_squared"])
    assert set(X_daily_order_2_const) == temp_expected
    temp_expected = set(expected_hourly_fourier_features + expected_hourly_lags)
    assert set(X_hourly_order_0_no_const) == temp_expected
    temp_expected = set(expected_hourly_fourier_features + expected_hourly_lags + ["const"])
    assert set(X_hourly_order_0_const) == temp_expected
    temp_expected = set(expected_hourly_fourier_features + expected_hourly_lags + ["trend", "trend_squared"])
    assert set(X_hourly_order_2_no_const) == temp_expected
    temp_expected = set(expected_hourly_fourier_features + expected_hourly_lags + ["const", "trend", "trend_squared"])
    assert set(X_hourly_order_2_const) == temp_expected

    # Check that the missing row was filled in correctly, note all original values are >= 1 so by checking its now zero we can see was filled in correctly
    assert y_daily_order_0_no_const.iloc[-2] == 0
    assert y_daily_order_0_const.iloc[-2] == 0
    assert y_daily_order_2_no_const.iloc[-2] == 0
    assert y_daily_order_2_const.iloc[-2] == 0
    assert y_hourly_order_0_no_const.iloc[-2] == 0
    assert y_hourly_order_0_const.iloc[-2] == 0
    assert y_hourly_order_2_no_const.iloc[-2] == 0
    assert y_hourly_order_2_const.iloc[-2] == 0  

    # Check that the lags are correctly returned
    assert lags_daily_order_0_no_const == lags_daily_order_0_const == lags_daily_order_2_no_const == lags_daily_order_2_const == daily_lags
    assert lags_hourly_order_0_no_const == lags_hourly_order_0_const == lags_hourly_order_2_no_const == lags_hourly_order_2_const == hourly_lags

    # Check that both the lags and fourier features have been implemented correctly
    assert check_lags_and_fourier_preprocess(dp_daily_order_0_no_const, X_daily_order_0_no_const, daily_lags, series_daily_full, "D") == True
    assert check_lags_and_fourier_preprocess(dp_daily_order_0_const, X_daily_order_0_const, daily_lags, series_daily_full, "D") == True
    assert check_lags_and_fourier_preprocess(dp_daily_order_2_no_const, X_daily_order_2_no_const, daily_lags, series_daily_full, "D") == True
    assert check_lags_and_fourier_preprocess(dp_daily_order_2_const, X_daily_order_2_const, daily_lags, series_daily_full, "D") == True
    assert check_lags_and_fourier_preprocess(dp_hourly_order_0_no_const, X_hourly_order_0_no_const, hourly_lags, series_hourly_full, "h") == True
    assert check_lags_and_fourier_preprocess(dp_hourly_order_0_const, X_hourly_order_0_const, hourly_lags, series_hourly_full, "h") == True
    assert check_lags_and_fourier_preprocess(dp_hourly_order_2_no_const, X_hourly_order_2_no_const, hourly_lags, series_hourly_full, "h") == True
    assert check_lags_and_fourier_preprocess(dp_hourly_order_2_const, X_hourly_order_2_const, hourly_lags, series_hourly_full, "h") == True

def test_to_NYC():
    """ test for to_NYC function in forecast_helpers.py 

    the series are structure to test that we are filling in the gaps in the series according to the time step, this is actually a fail safe
    anyway as in our own series all gaps should be filled during processing.
    """    
    import pandas as pd
    from pandas import Timestamp
    from jfk_taxis import forecast_helpers 

    series_hourly = pd.Series(
        data = [200, 147, 23],
        index = [Timestamp("2011-05-25 18:00:00+00:00", tz = "UTC"), Timestamp("2011-05-25 19:00:00+00:00", tz = "UTC"), Timestamp("2011-05-25 21:00:00+00:00", tz = "UTC")]
    )

    converted_series = forecast_helpers.to_NYC(series_hourly, "h")

    assert converted_series.index.dtype == "datetime64[ns, America/New_York]"
    assert converted_series.index[0] == Timestamp("2011-05-25 14:00:00-0400", tz='America/New_York')  # UTC-4
    assert converted_series.index[1] == Timestamp("2011-05-25 15:00:00-0400", tz='America/New_York')  # UTC-4
    assert converted_series.index[2] == Timestamp("2011-05-25 16:00:00-0400", tz='America/New_York')  # UTC-4
    assert converted_series.index[3] == Timestamp("2011-05-25 17:00:00-0400", tz='America/New_York')  # UTC-4

    series_daily = pd.Series(
        data = [242, 2, 1],
        index = [Timestamp("2024-01-01 00:00:00+00:00", tz = "UTC"), Timestamp("2024-01-03 00:00:00+00:00", tz = "UTC"), Timestamp("2024-01-04 00:00:00+00:00", tz = "UTC")]
    )

    converted_series = forecast_helpers.to_NYC(series_daily, "D")

    assert converted_series.index.dtype == "datetime64[ns, America/New_York]"
    assert converted_series.index[0] == Timestamp("2023-12-31 19:00:00-0500", tz='America/New_York')  
    assert converted_series.index[1] == Timestamp("2024-01-01 19:00:00-0500", tz='America/New_York')  
    assert converted_series.index[2] == Timestamp("2024-01-02 19:00:00-0500", tz='America/New_York')  
    assert converted_series.index[3] == Timestamp("2024-01-03 19:00:00-0500", tz='America/New_York')  

def preprocess_model(model_type: str, time_step: str) -> tuple[pd.DataFrame, pd.Series, DeterministicProcess, list[int]]:
    """ function to generate some sample preprocessed data for testing the forecast function

    Args:
        model_type (str): type of model, linear, hybrid or non-linear
        time_step (str): time step, "D" for daily, "h" for hourly

    Returns:
        tuple[pd.DataFrame, pd.Series, DeterministicProcess, list[int]]: X, y, dp, lags
    """     
    import pandas as pd
    import numpy as np
    from jfk_taxis import forecast_helpers

    # Sample time series
    np.random.seed(37)
    series_daily_full = pd.Series(data = np.random.uniform(3500, 5500, size = 365*3), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3, freq = "D"))
    series_hourly_full = pd.Series(data = np.random.uniform(50, 400, size = 365*3*24), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3*24, freq = "h"))

    # Lags
    daily_lags = [1, 2, 7, 23, 364]
    hourly_lags = [1, 2, 24, 48, 24*365+12]

    # Order
    order = 2 # we will just use order 2 for all models here
    constant = True

    # Fourier features
    daily_fourier = ["YE", "W"]
    hourly_fourier = ["D", "W"]

    if time_step == "D":
        if model_type == "non-linear":
            constant = False
            order = 0

        return forecast_helpers.preprocess(daily_lags, constant = constant, order = order, fourier_features = daily_fourier, time_step = "D", ts = series_daily_full)
    elif time_step == "h":
        if model_type == "non-linear":
            constant = False
            order = 0

        return forecast_helpers.preprocess(hourly_lags, constant = constant, order = order, fourier_features = hourly_fourier, time_step = "h", ts = series_hourly_full)

def test_forecast():
    """ test for forecast function in forecast_helpers.py
    """     
    import pandas as pd
    import numpy as np
    from jfk_taxis import forecast_helpers
    from jfk_taxis import load_config

    config, PROJECT_ROOT = load_config()

    # First we are going to need to create six models to test with, one linear, one hybrid and one non-linear for both hourly and daily time steps 
    # To do this we will use the prepocess function as that has been tested above, and this is also what the forecast function
    # will be recieveing as input anyway.

    model_types = ["linear", "hybrid", "non-linear"]
    time_steps = ["D", "h"]
    daily_steps = 30
    hourly_steps = 7*24

    for model_type in model_types:
        for time_step in time_steps:
            X, y, dp, lags = preprocess_model(model_type, time_step)

            # Fit the model
            if model_type == "linear":
                model = forecast_helpers.fit_linear(X, y)
                hybrid = None
                gpu = False
            elif model_type == "non-linear":
                model = forecast_helpers.fit_non_linear(X, y)
                hybrid = None
                if config["xgboost_setup"]["device"] == "cuda":
                    gpu = True
                else:
                    gpu = False
            elif model_type == "hybrid":
                model = forecast_helpers.fit_linear(X, y)
                X_num = X.to_numpy()
                residuals = y - pd.Series(data = model.predict(X_num), index = y.index)
                hybrid = forecast_helpers.fit_non_linear(X, residuals)
                gpu = False # as we are using a linear model as well we will just set gpu to False
                
            else:
                raise ValueError(f"Unknown model type {model_type}")

            if time_step == "D":
                steps = daily_steps
                offset = 11
            elif time_step == "h":
                steps = hourly_steps
                offset = 28
            else:
                raise ValueError(f"Unknown time step {time_step}") 
            


            forecast_series = forecast_helpers.forecast(model, y, lags, steps, offset, dp, hybrid, gpu)

            # Basic checks
            assert isinstance(forecast_series, pd.Series)
            assert forecast_series.shape[0] == steps
            assert forecast_series.index.dtype == "datetime64[ns]"

            # Check start end are correct and that each prediction has time step of "time_step" apart 
            assert forecast_series.index[0] == y.index[-1] + np.timedelta64((offset+1), time_step)
            assert forecast_series.index[-1] == y.index[-1] + np.timedelta64((offset + steps), time_step)
            for i in range(0, forecast_series.shape[0]-1):
                assert forecast_series.index[i+1] == forecast_series.index[i] + np.timedelta64(1, time_step)
            
            # Check all values are floats
            assert all([isinstance(val, float) for val in forecast_series])

def test_average_mae_by_step():
    """ test for the average_mae_by_step function from the ModelMAEScores class in forecast_helpers.py
    """    
    import pandas as pd
    from jfk_taxis import forecast_helpers

    # Sample maes
    maes = [200.3, 2424, 123.4, 543.2, 654.1, 234.5, 876.3, 345.2, 765.4, 234.1]
    # Sample steps
    steps = [1, 2, 3, 27, 1, 1, 2, 3, 1, 3]
    # Sample offsets
    offsets = [0, 0, 0, 0, 0, 10, 10, 20, 20, 20]
     

    # Create a sample ModelMAEScores object
    model_mae_scores = forecast_helpers.ModelMAEScores("test_model")

    for mae, step, offset in zip(maes, steps, offsets):
        score_obj = forecast_helpers.MAEScore(name="test_model", mae=mae, step=step, offset=offset)
        model_mae_scores.append_score(score_obj)

    avg_mae_by_step = model_mae_scores.average_mae_by_step()

    avg_by_step_1 = (200.3 + 654.1 + 234.5 + 765.4) / 4
    avg_mae_by_step_2 = (2424 + 876.3) / 2
    avg_mae_by_step_3 = (123.4 + 345.2 + 234.1) / 3
    avg_mae_by_step_27 = 543.2

    print(avg_mae_by_step.keys()) 
    # Checks
    assert avg_mae_by_step[1] == avg_by_step_1
    assert avg_mae_by_step[2] == avg_mae_by_step_2
    assert avg_mae_by_step[3] == avg_mae_by_step_3
    assert avg_mae_by_step[27] == avg_mae_by_step_27

def test_save_mae_scores():
    """ test for save_mae_scores function in forecast_helpers.py
    """   
    import pandas as pd
    from jfk_taxis import forecast_helpers

    # Sample data 
    model_mae_list = {
        "model_1": forecast_helpers.ModelMAEScores("model_1"),
        "model_2": forecast_helpers.ModelMAEScores("model_2"),
        "model_3": forecast_helpers.ModelMAEScores("model_3")  
    }

    mae_scores = {
        "model_1": 245.6,
        "model_2": 123.4,
        "model_3": 543.2
    }

    step = 5
    offset = 10

    model_mae_list = forecast_helpers.save_mae_scores(model_mae_list, mae_scores, step, offset)

    # Checks
    assert len(model_mae_list["model_1"].scores) == 1
    assert len(model_mae_list["model_2"].scores) == 1
    assert len(model_mae_list["model_3"].scores) == 1
    assert model_mae_list["model_1"].scores[0].mae == 245.6
    assert model_mae_list["model_2"].scores[0].mae == 123.4
    assert model_mae_list["model_3"].scores[0].mae == 543.2
    assert model_mae_list["model_1"].scores[0].step == step
    assert model_mae_list["model_2"].scores[0].step == step
    assert model_mae_list["model_3"].scores[0].step == step
    assert model_mae_list["model_1"].scores[0].offset == offset
    assert model_mae_list["model_2"].scores[0].offset == offset
    assert model_mae_list["model_3"].scores[0].offset == offset


# To debug tests run:
# 

def test_create_avg_mae_df():
    """ test for create_avg_mae_df function in forecast_helpers.py, we will reuse the save_mae_scores function from the above test as we know it works now.
    """    
    import pandas as pd
    from jfk_taxis import forecast_helpers

    # Sample data
    model_mae_list = {
        "model_1": forecast_helpers.ModelMAEScores("model_1"),
        "model_2": forecast_helpers.ModelMAEScores("model_2"),
        "model_3": forecast_helpers.ModelMAEScores("model_3"),
        "Naive": forecast_helpers.ModelMAEScores("Naive")  
    }

    mae_scores = {
        "model_1": [245.6, 234.5, 210.3],
        "model_2": [123.4, 130.2, 125.6],
        "model_3": [543.2, 532.1, 550.3],
        "Naive": [300.1, 310.2, 305.3]
    }

    # These dicts don't need real values only the keys are used by the function
    linear_models = {
        "model_1": "tmp_str" 
    }

    non_linear_models = {
        "model_2": "tmp_str", 
        "model_3": "tmp3_str" 
    }

    # We will use naive 
    naive = True

    step = 5
    offset = 12

    # Save these set of scores and offsets
    for i in range(0, len(mae_scores["model_1"])):
        temp_mae_scores = {key: mae_scores[key][i] for key in mae_scores.keys()}
        model_mae_list = forecast_helpers.save_mae_scores(model_mae_list, temp_mae_scores, step, offset)

    # Now change the offset and step and save (you have to change the step each time as that is how we have setup forecast_helpers)
    mae_scores = {
        "model_1": [220.4, 215.6],
        "model_2": [128.9, 119.5],
        "model_3": [520.5, 530.2],
        "Naive": [315.4, 320.1]
    }
    offset = 20
    step = 7
    for i in range(0, len(mae_scores["model_1"])):
        temp_mae_scores = {key: mae_scores[key][i] for key in mae_scores.keys()}
        model_mae_list = forecast_helpers.save_mae_scores(model_mae_list, temp_mae_scores, step, offset)

    # Change just the step and save
    mae_scores = {
        "model_1": [200.3, 210.4],
        "model_2": [121.4, 119.8],
        "model_3": [510.2, 500.3],
        "Naive": [290.3, 295.4]
    }
    step = 10
    offset = 12
    for i in range(0, len(mae_scores["model_1"])):
        temp_mae_scores = {key: mae_scores[key][i] for key in mae_scores.keys()}
        model_mae_list = forecast_helpers.save_mae_scores(model_mae_list, temp_mae_scores, step, offset)

    # Change both and save
    mae_scores = {
        "model_1": [190.2, 525.3],
        "model_2": [115.6, 120.4],
        "model_3": [480.5, 490.2],
        "Naive": [280.2, 285.3]
    }
    offset = 20
    step = 42
    for i in range(0, len(mae_scores["model_1"])):
        temp_mae_scores = {key: mae_scores[key][i] for key in mae_scores.keys()}
        model_mae_list = forecast_helpers.save_mae_scores(model_mae_list, temp_mae_scores, step, offset)

    avg_mae_df = forecast_helpers.create_avg_mae_df(model_mae_list, linear_models, non_linear_models, naive)

    # Checks
    print(avg_mae_df)
    assert isinstance(avg_mae_df, pd.DataFrame)
    assert avg_mae_df.shape[0] == 4 # 4 different steps
    assert avg_mae_df.shape[1] == 4 # model_1, model_2, model_3, Naive
    assert set(list(avg_mae_df.index)) == {5, 7, 10, 42}
    assert set(list(avg_mae_df.columns)) == {"model_1", "model_2", "model_3", "Naive"}

    # Check the values
    assert avg_mae_df.loc[5, "model_1"] == (245.6 + 234.5 + 210.3) / 3
    assert avg_mae_df.loc[5, "model_2"] == (123.4 + 130.2 + 125.6) / 3
    assert avg_mae_df.loc[5, "model_3"] == (543.2 + 532.1 + 550.3) / 3
    assert avg_mae_df.loc[5, "Naive"] == (300.1 + 310.2 + 305.3) / 3
    assert avg_mae_df.loc[7, "model_1"] == (220.4 + 215.6) / 2
    assert avg_mae_df.loc[7, "model_2"] == (128.9 + 119.5) / 2
    assert avg_mae_df.loc[7, "model_3"] == (520.5 + 530.2) / 2
    assert avg_mae_df.loc[7, "Naive"] == (315.4 + 320.1) / 2
    assert avg_mae_df.loc[10, "model_1"] == (200.3 + 210.4) / 2
    assert avg_mae_df.loc[10, "model_2"] == (121.4 + 119.8) / 2
    assert avg_mae_df.loc[10, "model_3"] == (510.2 + 500.3) / 2
    assert avg_mae_df.loc[10, "Naive"] == (290.3 + 295.4) / 2
    assert avg_mae_df.loc[42, "model_1"] == (190.2 + 525.3) / 2
    assert avg_mae_df.loc[42, "model_2"] == (115.6 + 120.4) / 2
    assert avg_mae_df.loc[42, "model_3"] == (480.5 + 490.2) / 2
    assert avg_mae_df.loc[42, "Naive"] == (280.2 + 285.3) / 2



    






