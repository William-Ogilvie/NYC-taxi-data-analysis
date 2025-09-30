"""
test_training_helpers.py
=========================

Unit test for training_helpers.py.
"""   

def test_load_ts_data():
    """ test the load_ts_data function in training_helpers.py, note this test will only pass if there is actual some time series data to load.
    So to run the test properly you will first need to have run the data processing notebook. We have also checked manually that the time series loaded
    from this function have the correct values etc. This test is more like a sanity check.
    """        
    from jfk_taxis import training_helpers
    import pandas as pd


    # Load data
    ts_daily, ts_hourly = training_helpers.load_ts_data()

    # Check types
    assert isinstance(ts_daily, pd.Series), "ts_daily is not a pd.Series"
    assert isinstance(ts_hourly, pd.Series), "ts_hourly is not a pd.Series"
    assert isinstance(ts_daily.index, pd.DatetimeIndex), "ts_daily index is not a pd.DatetimeIndex"
    assert isinstance(ts_hourly.index, pd.DatetimeIndex), "ts_hourly index is not a pd.DatetimeIndex"

    # Check time step
    for i in range(0, len(ts_daily) - 1):
        assert (ts_daily.index[i+1] - ts_daily.index[i]).days == 1, "ts_daily does not have daily frequency"
    for i in range(0, len(ts_hourly) - 1):
        assert (ts_hourly.index[i+1] - ts_hourly.index[i]).seconds == 3600, "ts_hourly does not have hourly frequency"

def test_load_ts_data_app():
    """ test the load_ts_data_app function in training_helpers.py, note this test will only pass if there is actual some time series data to load for the app.
    """    
    from jfk_taxis import training_helpers
    import pandas as pd


    # Load data
    ts_daily, ts_hourly = training_helpers.load_ts_data_app()

    # Check types
    assert isinstance(ts_daily, pd.Series), "ts_daily is not a pd.Series"
    assert isinstance(ts_hourly, pd.Series), "ts_hourly is not a pd.Series"
    assert isinstance(ts_daily.index, pd.DatetimeIndex), "ts_daily index is not a pd.DatetimeIndex"
    assert isinstance(ts_hourly.index, pd.DatetimeIndex), "ts_hourly index is not a pd.DatetimeIndex"

    # Check time step
    for i in range(0, len(ts_daily) - 1):
        assert (ts_daily.index[i+1] - ts_daily.index[i]).days == 1, "ts_daily does not have daily frequency"
    for i in range(0, len(ts_hourly) - 1):
        assert (ts_hourly.index[i+1] - ts_hourly.index[i]).seconds == 3600, "ts_hourly does not have hourly frequency"

def test_split_test_train_sets():
    """ Test the split_test_train_sets function in training_helpers.py
    """    
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    import pandas as pd

    # First we will load the data using load_ts_data (which has been tested above)
    ts_daily, ts_hourly = training_helpers.load_ts_data()

    # Now we will split the data into test and train sets, on the boundary specified in the config file
    config, PROJECT_ROOT =  load_config()

    ts_daily_train, ts_daily_test, ts_hourly_train, ts_hourly_test = training_helpers.split_test_train_sets(ts_daily, ts_hourly)

    # Check types
    assert isinstance(ts_daily_train, pd.Series), "ts_daily_train is not a pd.Series"
    assert isinstance(ts_daily_test, pd.Series), "ts_daily_test is not a pd.Series"
    assert isinstance(ts_hourly_train, pd.Series), "ts_hourly_train is not a pd.Series" 
    assert isinstance(ts_hourly_test, pd.Series), "ts_hourly_test is not a pd.Series"
    assert isinstance(ts_daily_train.index, pd.DatetimeIndex), "ts_daily_train index is not a pd.DatetimeIndex"
    assert isinstance(ts_daily_test.index, pd.DatetimeIndex), "ts_daily_test index is not a pd.DatetimeIndex"
    assert isinstance(ts_hourly_train.index, pd.DatetimeIndex), "ts_hourly_train index is not a pd.DatetimeIndex"
    assert isinstance(ts_hourly_test.index, pd.DatetimeIndex), "ts_hourly_test index is not a pd.DatetimeIndex"

    # Check that the series have correct time steps
    for i in range(0, len(ts_daily_train) - 1):
        assert (ts_daily_train.index[i+1] - ts_daily_train.index[i]).days == 1, "ts_daily_train does not have daily frequency"
    for i in range(0, len(ts_daily_test) - 1):
        assert (ts_daily_test.index[i+1] - ts_daily_test.index[i]).days == 1, "ts_daily_test does not have daily frequency"  
    for i in range(0, len(ts_hourly_train) - 1):
        assert (ts_hourly_train.index[i+1] - ts_hourly_train.index[i]).seconds == 3600, "ts_hourly_train does not have hourly frequency"
    for i in range(0, len(ts_hourly_test) - 1):
        assert (ts_hourly_test.index[i+1] - ts_hourly_test.index[i]).seconds == 3600, "ts_hourly_test does not have hourly frequency"

    # Check that the split is correct
    assert ts_daily_train.index[-1].tz_localize(None) == pd.to_datetime(config["modelling"]["ts_daily_train_boundary"]), "ts_daily_train does not end on the correct date"
    assert ts_daily_test.index[0].tz_localize(None) == pd.to_datetime(config["modelling"]["ts_daily_test_boundary"]), "ts_daily_test does not start on the correct date"
    assert ts_hourly_train.index[-1] == pd.to_datetime(config["modelling"]["ts_hourly_train_boundary"]), "ts_hourly_train does not end on the correct date"
    assert ts_hourly_test.index[0] == pd.to_datetime(config["modelling"]["ts_hourly_test_boundary"]), "ts_hourly_test does not start on the correct date"

    # Sanity check that the boundaries are correct themselves
    assert pd.to_datetime(config["modelling"]["ts_daily_train_boundary"]) == pd.to_datetime(config["modelling"]["ts_daily_test_boundary"]) - pd.Timedelta(days=1), "ts_daily_train_boundary is not one day before ts_daily_test_boundary"
    assert pd.to_datetime(config["modelling"]["ts_hourly_train_boundary"]) == pd.to_datetime(config["modelling"]["ts_hourly_test_boundary"]) - pd.Timedelta(hours=1), "ts_hourly_train_boundary is not one hour before ts_hourly_test_boundary"