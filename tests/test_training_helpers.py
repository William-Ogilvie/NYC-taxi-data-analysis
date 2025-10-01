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

def test_load_process_lags():
    """ test load_process_lags function in training_helpers.py
    """    
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config

    # Load config and the lags we expect
    config, PROJECT_ROOT =  load_config()

    expected_daily_lags = config["modelling"]["daily_lags"]
    expected_hourly_lags = config["modelling"]["hourly_lags"]
    expected_used_hourly_lags = expected_hourly_lags[:config["modelling"]["hourly_num_lags"]]
    expected_extra_hourly_lags = config["modelling"]["hourly_extra_lags"]

    daily_lags, used_hourly_lags = training_helpers.load_process_lags()

    # Check that the lags are correct
    assert daily_lags == expected_daily_lags, f"daily_lags {daily_lags} does not match expected {expected_daily_lags}"
    assert len(used_hourly_lags) == len(expected_used_hourly_lags) + len(expected_extra_hourly_lags), f"used_hourly_lags {used_hourly_lags} does not have the correct length"
    for lag in used_hourly_lags:
        assert lag in expected_used_hourly_lags + expected_extra_hourly_lags, f"lag {lag} in used_hourly_lags {used_hourly_lags} is not in expected {expected_used_hourly_lags + expected_extra_hourly_lags}"

def test_save_models():
    """ test save_models function in training_helpers.py
    """    
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from statsmodels.tsa.deterministic import DeterministicProcess
    from test_modelling_helpers import create_ts

    # Load config
    config, PROJECT_ROOT =  load_config()

    # We will use a daily time series for the first model and an hourly one for the second
    ts_daily = create_ts("D")
    ts_hourly = create_ts("h")

    # To test that the save models function works we will create some dummy models, and save them and check they exist in the correct form in the right directory
    # we won't actually load them back in this test as that will be done in the test_load_models function

    # We create two dummy linear models (one hybrid and one not) and two dummy non-linear models 
    dummy_linear_model = LinearRegression()  
    dummy_hybrid_component = XGBRegressor()
    dummy_dp_linear_daily = DeterministicProcess(index = ts_daily.index, order=3, seasonal=True, constant = True)
    dummy_dp_linear_hourly = DeterministicProcess(index = ts_hourly.index, order=3, seasonal=True, constant = True) 
    dummy_lags_linear = [1,2,3,4,5]
    dummy_lags_linear_2 = [1,2,3]
    
    dummmy_non_linear_model = XGBRegressor()
    dummy_non_linear_hybrid = None
    dummy_dp_non_linear_daily = DeterministicProcess(index = ts_daily.index, constant = False, order = 0, seasonal = True)
    dummy_dp_non_linear_hourly = DeterministicProcess(index = ts_hourly.index, constant = False, order = 0, seasonal = True) 
    dummy_lags_non_linear = [1,2,3,4,5,6,7,8,9,10]
    dummy_lags_non_linear_2 = [1,2,3,4,5]

    linear_model_names = ["test_linear_model", "test_linear_model_2"]
    non_linear_model_names = ["test_non_linear_model", "test_non_linear_model_2"]


    linear_models = {
        linear_model_names[0]: (dummy_linear_model, dummy_dp_linear_daily, dummy_hybrid_component, dummy_lags_linear), 
        linear_model_names[1]: (dummy_linear_model, dummy_dp_linear_hourly, None, dummy_lags_linear_2) 
    }

    non_linear_models = {
        non_linear_model_names[0]: (dummmy_non_linear_model, dummy_dp_non_linear_daily, dummy_non_linear_hybrid, dummy_lags_non_linear),
        non_linear_model_names[1]: (dummmy_non_linear_model, dummy_dp_non_linear_hourly, dummy_non_linear_hybrid, dummy_lags_non_linear_2)
    }

    # Save the models
    sig = "test_model_sig"
    training_helpers.save_models(linear_models, non_linear_models, sig)

    # Now check that the models exist in the correct directory
    SAVED_OBJECTS_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["saved_objects_path"]

    # Saving constants
    MODEL = config["saving"]["model_file_suffix"]
    HYBRID = config["saving"]["hybrid_file_suffix"]
    DP = config["saving"]["dp_file_suffix"]
    LAGS =  config["saving"]["lags_file_suffix"]

    expected_files = [
        SAVED_OBJECTS_DIR / f"{name}_{sig}_{suffix}"
        for name in linear_model_names + non_linear_model_names
        for suffix in [MODEL, HYBRID, DP, LAGS]
    ]

    # Check that the files exist
    for f in expected_files:
        assert f.exists(), f"Expected file {f} does not exist"

def test_save_design():
    """ test the save_design function in training_helpers.py
    """    
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    from test_modelling_helpers import create_ts
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.deterministic import DeterministicProcess

    # Load config
    config, PROJECT_ROOT =  load_config()

    # We will use a daily time series for the first design and an hourly one for the second
    ts_daily = create_ts("D")
    ts_hourly = create_ts("h")

    # Daily case

    # Create dummy designs, lags and dp's, the ts above are the targets
    np.random.seed(37)
    n = len(ts_daily) # Number of rows
    columns = ["sin(1,freq=YE-DEC)", "lag_6", "trend", "cos(3,freq=D)", "lag_15"]
    dummy_X_daily_1 = pd.DataFrame(np.random.rand(n, len(columns)), columns = columns) 
    dummy_X_daily_2 = pd.DataFrame(np.random.rand(n, len(columns)), columns = columns)
    dp_daily_1 = DeterministicProcess(index = ts_daily.index, order = 3, constant = True)
    dp_daily_2 = DeterministicProcess(index = ts_daily.index, order = 0, seasonal=True, constant = True) 
    lags_daily_1 = [1,2,3,4,5,6]
    lags_daily_2 = [1, 2, 3]
    
    linear_design_names = ["test_linear_design", "test_linear_design_2"]
    non_linear_design_names = ["test_non_linear_design", "test_non_linear_design_2"]


    linear_design = {
        linear_design_names[0]: (dummy_X_daily_1, ts_daily, dp_daily_1, lags_daily_1), 
        linear_design_names[1]: (dummy_X_daily_2, ts_daily, dp_daily_2, lags_daily_2) 
    }

    non_linear_design = {
        non_linear_design_names[0]: (dummy_X_daily_1, ts_daily, dp_daily_1, lags_daily_1),
        non_linear_design_names[1]: (dummy_X_daily_2, ts_daily, dp_daily_2, lags_daily_2)
    }

    # Save the models
    sig = "test_model_sig"
    training_helpers.save_design(linear_design, non_linear_design, sig)

    # Now check that the models exist in the correct directory
    SAVED_OBJECTS_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["saved_objects_path"]

    # Saving constants
    DESIGN = config["saving"]["design_file_suffix"]
    TARGET = config["saving"]["target_file_suffix"]
    DP = config["saving"]["dp_file_suffix"]
    LAGS =  config["saving"]["lags_file_suffix"]

    expected_files = [
        SAVED_OBJECTS_DIR / f"{name}_{sig}_{suffix}"
        for name in linear_design_names + non_linear_design_names
        for suffix in [DESIGN, TARGET, DP, LAGS]
    ]

    # Check that the files exist
    for f in expected_files:
        assert f.exists(), f"Expected file {f} does not exist"



