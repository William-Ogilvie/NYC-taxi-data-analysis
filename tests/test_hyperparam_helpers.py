"""
test_hyperparam_helpers.py
============================

Unit tests for hyperparam_helpers.py
"""

def test_create_val_data():
    """ test for create_val_data function in hyperparam_helpers.py, note we don't need to test the output of the 
    preprocessing as this is tested in forecast_helpers.py, we just need to check that the correct slices of the data have been taken.
    """    
    import pandas as pd
    import numpy as np
    from jfk_taxis import hyperparam_helpers

    # We will do two sets of sample data one with daily data and one with hourly data 
    series_daily_full = pd.Series(data = np.random.uniform(3500, 5500, size = 365*3), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3, freq = "D"))
    series_hourly_full = pd.Series(data = np.random.uniform(50, 400, size = 365*3*24), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3*24, freq = "h"))

    n_splits = 5

    daily_test_size = 30
    hourly_test_size = 24*7

    # Constant and order are less important as in the preprocessing test we have verified these work correctly
    constant = True
    order = 1

    # Similarly with lags and fourier features
    daily_lags = [1, 2, 7, 364]
    hourly_lags = [1, 2, 24, 168]

    daily_fourier_features = ["YE", "W"]
    hourly_fourier_features = ["W", "D"]

    daily_fold_dict = hyperparam_helpers.create_val_data(n_splits, daily_test_size, daily_lags, constant, order, daily_fourier_features, "D", series_daily_full)
    hourly_fold_dict = hyperparam_helpers.create_val_data(n_splits, hourly_test_size, hourly_lags, constant, order, hourly_fourier_features, "h", series_hourly_full)

    # The primary thing we need to check is that the correct slices of the data have been taken
    # We will do this by checking that first y_test is the correct size and then that y_test is a continuation of y_train 
    for key, value in daily_fold_dict.items():
        y_train = value[1]
        y_test = value[4]

        # Length of y_test
        assert len(y_test) == daily_test_size

        # y_test 1 time step after y_train
        assert y_test.index[0].tz_localize(None) == y_train.index[-1] +  np.timedelta64(1, "D")

        # y_test formatted correctly (this isn't given as y_test doesn't come from preprocesss so worth checking)
        for i in range(0, y_test.shape[0]-1):
            assert y_test.index[i+1] == y_test.index[i] + np.timedelta64(1, "D")

    for key, value in hourly_fold_dict.items():
        y_train = value[1]
        y_test = value[4]

        # Length of y_test
        assert len(y_test) == hourly_test_size

        # y_test 1 time step after y_train
        assert y_test.index[0].tz_localize(None) == y_train.index[-1] +  np.timedelta64(1, "h")

        # y_test formatted correctly (this isn't given as y_test doesn't come from preprocesss so worth checking)
        for i in range(0, y_test.shape[0]-1):
            assert y_test.index[i+1] == y_test.index[i] + np.timedelta64(1, "h")


