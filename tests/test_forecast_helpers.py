"""
test_forecast_helpers.py
=========================

Unit tests for forecast helpers.py.
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
                                      Timestamp("2011-12-31 23:00:00")]))



def check_lags_preprocess(dp: DeterministicProcess, X: pd.DataFrame, lags: list[int], y: pd.Series) -> bool:
    """ checks that the lags implemented by the preprocess have been done correctly

    Args:
        dp (DeterministicProcess): deterministic process in question
        X (pd.DataFrame): design matrix
        lags (list[int]): list of lags to check
        y (pd.Series): original time series

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
    step = 30
    X_out = dp.out_of_sample(steps = step)
    for i in range(0, X_out.shape[0]):
        print(X_out.iloc[i])


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

    # We are going to trust that deterministic process from statsmodels is implemented correctly on their end so we only need to check that our lags are done correctly
    # using the check_lags_preprocess function
    assert check_lags_preprocess(dp_daily_order_0_no_const, X_daily_order_0_no_const, daily_lags, series_daily_full) == True
    assert check_lags_preprocess(dp_daily_order_0_const, X_daily_order_0_const, daily_lags, series_daily_full) == True
    assert check_lags_preprocess(dp_daily_order_2_no_const, X_daily_order_2_no_const, daily_lags, series_daily_full) == True
    assert check_lags_preprocess(dp_daily_order_2_const, X_daily_order_2_const, daily_lags, series_daily_full) == True
    assert check_lags_preprocess(dp_hourly_order_0_no_const, X_hourly_order_0_no_const, hourly_lags, series_hourly_full) == True
    assert check_lags_preprocess(dp_hourly_order_0_const, X_hourly_order_0_const, hourly_lags, series_hourly_full) == True
    assert check_lags_preprocess(dp_hourly_order_2_no_const, X_hourly_order_2_no_const, hourly_lags, series_hourly_full) == True
    assert check_lags_preprocess(dp_hourly_order_2_const, X_hourly_order_2_const, hourly_lags, series_hourly_full) == True



     