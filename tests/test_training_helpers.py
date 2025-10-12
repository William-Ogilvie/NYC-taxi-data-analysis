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

# def test_load_ts_data_app():
#     """ test the load_ts_data_app function in training_helpers.py, note this test will only pass if there is actual some time series data to load for the app.
#     """    
#     from jfk_taxis import training_helpers
#     import pandas as pd


#     # Load data
#     ts_daily, ts_hourly = training_helpers.load_ts_data_app()

#     # Check types
#     assert isinstance(ts_daily, pd.Series), "ts_daily is not a pd.Series"
#     assert isinstance(ts_hourly, pd.Series), "ts_hourly is not a pd.Series"
#     assert isinstance(ts_daily.index, pd.DatetimeIndex), "ts_daily index is not a pd.DatetimeIndex"
#     assert isinstance(ts_hourly.index, pd.DatetimeIndex), "ts_hourly index is not a pd.DatetimeIndex"

#     # Check time step
#     for i in range(0, len(ts_daily) - 1):
#         assert (ts_daily.index[i+1] - ts_daily.index[i]).days == 1, "ts_daily does not have daily frequency"
#     for i in range(0, len(ts_hourly) - 1):
#         assert (ts_hourly.index[i+1] - ts_hourly.index[i]).seconds == 3600, "ts_hourly does not have hourly frequency"

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
    from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

    # Load config
    config, PROJECT_ROOT =  load_config()

    # We will use a daily time series for the first design and an hourly one for the second
    ts_daily = create_ts("D")
    ts_hourly = create_ts("h")

    # We need to remove the time zone so we can add fourier features
    ts_daily.index = ts_daily.index.tz_localize(None)
    ts_hourly.index = ts_hourly.index.tz_localize(None)

    # We also need to reset the frequencies
    ts_daily = ts_daily.asfreq("D").fillna(0)
    ts_hourly = ts_hourly.asfreq("h").fillna(0)

    # Create dummy designs, lags and dp's, the ts above are the targets
    np.random.seed(37)

    # Fourier features
    daily_fourier = [CalendarFourier(freq = "YE", order = 10), CalendarFourier(freq = "W", order = 5)]
    hourly_fourier = [CalendarFourier(freq = "W", order = 5), CalendarFourier(freq = "D", order = 5)]

    n_daily = len(ts_daily) # Number of rows
    n_hourly = len(ts_hourly) # Number of rows
    dp_daily = DeterministicProcess(index = ts_daily.index, order = 3, seasonal = False, constant = True, additional_terms = daily_fourier)
    dp_daily_2 = DeterministicProcess(index = ts_daily.index, order = 0, seasonal=False, constant = True, additional_terms = daily_fourier) 
    dp_hourly = DeterministicProcess(index = ts_hourly.index, order = 3, seasonal = False, constant = True, additional_terms = hourly_fourier)
    dp_hourly_2 = DeterministicProcess(index = ts_hourly.index, order = 0, seasonal = False, constant = False, additional_terms = hourly_fourier)
    

    dummy_X_daily_linear = dp_daily.in_sample()
    dummy_X_daily_non_linear = dp_daily_2.in_sample()
    dummy_X_hourly_linear = dp_hourly.in_sample()
    dummy_X_hourly_non_linear = dp_hourly_2.in_sample()

    lags_daily = [1,2,3,4,5,6]
    lags_hourly = [1, 2, 3, 480]
    
    linear_design_names = ["test_linear_design", "test_linear_design_2"]
    non_linear_design_names = ["test_non_linear_design", "test_non_linear_design_2"]


    linear_design = {
        linear_design_names[0]: (dummy_X_daily_linear, ts_daily, dp_daily, lags_daily), 
        linear_design_names[1]: (dummy_X_hourly_linear, ts_hourly, dp_hourly, lags_hourly) 
    }

    non_linear_design = {
        non_linear_design_names[0]: (dummy_X_daily_non_linear, ts_daily, dp_daily_2, lags_daily),
        non_linear_design_names[1]: (dummy_X_hourly_non_linear, ts_daily, dp_hourly_2, lags_hourly)
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

def test_load_models():
    """ test for load_models from training_helpers.py

        We will reuse the same setup for save models, we will then simply reload all the models and check they match what we think they should have saved as.
        This essentially tests two parts of the functionality, does the save_models function actually save the model correctly and does this load_models function load them correctly.
        It is easier to combine the tests this way, the test for save_models primarily tests that the files themselves look correct from the outside but doesn't actually look at the contents 
        of the files. 

        We will make some slight modifications in that we will now fit the models properly as well.
    """    
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
    from test_modelling_helpers import create_ts

    # Load config
    config, PROJECT_ROOT =  load_config()

    # We will use a daily time series for the first model and an hourly one for the second
    ts_daily = create_ts("D")
    ts_hourly = create_ts("h")

    # Create a test train split for the time series data
    ts_daily_train = ts_daily[:"2022-12-31"]
    ts_daily_test = ts_daily["2023-01-01":]

    ts_hourly_train = ts_hourly[:"2022-12-31 23:00:00+0000"]
    ts_hourly_test = ts_hourly["2023-01-01 00:00+0000":]

    # For passing to the deterministic process we will need to drop the time zone
    ts_daily_train.index = ts_daily_train.index.tz_localize(None)
    ts_daily_test.index = ts_daily_test.index.tz_localize(None)
    ts_hourly_train.index = ts_hourly_train.index.tz_localize(None)
    ts_hourly_test.index = ts_hourly_test.index.tz_localize(None)

    # We also need to set the frequency of the time series again
    ts_daily_train = ts_daily_train.asfreq("D").fillna(0)
    ts_daily_test = ts_daily_test.asfreq("D").fillna(0)
    ts_hourly_train = ts_hourly_train.asfreq("h").fillna(0)
    ts_hourly_test = ts_hourly_test.asfreq("h").fillna(0)
 

    # To test that the save models function works we will create some dummy models, and save them and check they exist in the correct form in the right directory
    # we won't actually load them back in this test as that will be done in the test_load_models function

    # We create two dummy linear models (one hybrid and one not) and two dummy non-linear models   
    dummy_linear_model_daily = LinearRegression()
    dummy_linear_model_hourly = LinearRegression()
    dummy_hybrid_component = XGBRegressor()
    daily_fourier = [CalendarFourier(freq = "YE", order = 10), CalendarFourier(freq = "W", order = 5)]
    hourly_fourier = [CalendarFourier(freq = "W", order = 5), CalendarFourier(freq = "D", order = 5)]
    dummy_dp_linear_daily = DeterministicProcess(index = ts_daily_train.index, order=3, seasonal=False, constant = True, additional_terms=daily_fourier)
    dummy_dp_linear_hourly = DeterministicProcess(index = ts_hourly_train.index, order=3, seasonal=False, constant = True, additional_terms=hourly_fourier) 
    dummy_lags_linear = [1,2,3,4,5]
    dummy_lags_linear_2 = [1,2,3]
    
    dummy_non_linear_model_daily = XGBRegressor()
    dummy_non_linear_model_hourly = XGBRegressor()
    dummy_non_linear_hybrid = None
    dummy_dp_non_linear_daily = DeterministicProcess(index = ts_daily_train.index, constant = False, order = 0, seasonal = False, additional_terms=daily_fourier)
    dummy_dp_non_linear_hourly = DeterministicProcess(index = ts_hourly_train.index, constant = False, order = 0, seasonal = False, additional_terms=hourly_fourier) 
    dummy_lags_non_linear = [1,2,3,4,5,6,7,8,9,10]
    dummy_lags_non_linear_2 = [1,2,3,4,5]

    linear_model_names = ["test_linear_model", "test_linear_model_2"]
    non_linear_model_names = ["test_non_linear_model", "test_non_linear_model_2"]

    # We are now going to fit the model to the design matricies
    X_linear_1 = dummy_dp_linear_daily.in_sample()
    dummy_linear_model_daily.fit(X_linear_1, ts_daily_train)
    y_fit_linear_1 = dummy_linear_model_daily.predict(X_linear_1)
    y_resid_1 = ts_daily_train - y_fit_linear_1  
    dummy_hybrid_component.fit(X_linear_1, y_resid_1)
    X_linear_2 = dummy_dp_linear_hourly.in_sample()
    dummy_linear_model_hourly.fit(X_linear_2, ts_hourly_train)

    X_non_linear_1 = dummy_dp_non_linear_daily.in_sample()
    dummy_non_linear_model_daily = dummy_non_linear_model_daily.fit(X_non_linear_1, ts_daily_train)
    X_non_linear_2 = dummy_dp_non_linear_hourly.in_sample()
    dummy_non_linear_model_hourly = dummy_non_linear_model_hourly.fit(X_non_linear_2, ts_hourly_train)

    linear_models = {
        linear_model_names[0]: (dummy_linear_model_daily, dummy_dp_linear_daily, dummy_hybrid_component, dummy_lags_linear), 
        linear_model_names[1]: (dummy_linear_model_hourly, dummy_dp_linear_hourly, None, dummy_lags_linear_2) 
    }

    non_linear_models = {
        non_linear_model_names[0]: (dummy_non_linear_model_daily, dummy_dp_non_linear_daily, dummy_non_linear_hybrid, dummy_lags_non_linear),
        non_linear_model_names[1]: (dummy_non_linear_model_hourly, dummy_dp_non_linear_hourly, dummy_non_linear_hybrid, dummy_lags_non_linear_2)
    }

    # Save the models
    sig = "test_model_sig"
    training_helpers.save_models(linear_models, non_linear_models, sig)

    # Reload the models
    linear_models_loaded, non_linear_models_loaded = training_helpers.load_models(sig)

    # Get test design matricies to use
    X_test_daily_linear = dummy_dp_linear_daily.out_of_sample(30)
    X_test_hourly_linear = dummy_dp_linear_hourly.out_of_sample(30)
    X_test_daily_non_linear = dummy_dp_non_linear_daily.out_of_sample(30)
    X_test_hourly_non_linear = dummy_dp_non_linear_hourly.out_of_sample(30)

    # Predictions to check against
    daily_preds_linear = dummy_linear_model_daily.predict(X_test_daily_linear)
    daily_preds_hybrid = dummy_hybrid_component.predict(X_test_daily_linear)
    hourly_preds_linear = dummy_linear_model_hourly.predict(X_test_hourly_linear)
    daily_preds_non_linear = dummy_non_linear_model_daily.predict(X_test_daily_non_linear)
    hourly_preds_non_linear = dummy_non_linear_model_hourly.predict(X_test_hourly_non_linear) 

    # Expected fourier terms
    expected_daily_fourier = [f"cos({x},freq=YE-DEC)" for x in range(1, 11)] + [f"sin({x},freq=YE-DEC)" for x in range(1, 11)] + \
                            [f"cos({x},freq=W-SUN)" for x in range(1, 6)] + [f"sin({x},freq=W-SUN)" for x in range(1, 6)]
    expected_hourly_fourier = [f"cos({x},freq=D)" for x in range(1, 6)] + [f"sin({x},freq=D)" for x in range(1, 6)] + \
                            [f"cos({x},freq=W-SUN)" for x in range(1, 6)] + [f"sin({x},freq=W-SUN)" for x in range(1, 6)]

    # Expected order terms
    expected_order_terms = ["const", "trend", "trend_squared", "trend_cubed"]

    # Checks
    assert len(linear_models_loaded.keys()) == len(linear_model_names), "linear_models does not have the same number of keys as length of linear_model_names"
    assert set(linear_models_loaded.keys()) == set(linear_model_names), "linear_models doesn't have the same keys as in linear_model_names"
    assert len(non_linear_models_loaded.keys()) == len(non_linear_model_names), "non_linear_models does not have the same number of keys as length of non_linear_model_names"
    assert set(non_linear_models_loaded.keys()) == set(non_linear_model_names), "non_linear_models does not have the same keys as in non_linear_model_names"
    assert linear_models_loaded[linear_model_names[0]][0].get_params() == dummy_linear_model_daily.get_params(), f"linear model params do not match for {linear_model_names[0]}" 
    assert all(linear_models_loaded[linear_model_names[0]][0].predict(X_test_daily_linear) ==  daily_preds_linear), f"model predictions do not match for {linear_model_names[0]}"
    assert len(linear_models_loaded[linear_model_names[0]][1].out_of_sample(10).columns.tolist()) == len(expected_daily_fourier + expected_order_terms), f"dp doesn't have correct number of columns for {linear_model_names[0]}"
    assert set(linear_models_loaded[linear_model_names[0]][1].out_of_sample(10).columns.tolist()) == set(expected_daily_fourier + expected_order_terms), f"dp doesn't match columns for {linear_model_names[0]}"
    assert set(linear_models_loaded[linear_model_names[0]][2].get_params()) == set(dummy_hybrid_component.get_params()), f"hybrid params do not match for {linear_model_names[0]}"
    assert all(linear_models_loaded[linear_model_names[0]][2].predict(X_test_daily_linear) == daily_preds_hybrid), f"hybrid predictions do not match for {linear_model_names[0]}" 
    assert linear_models_loaded[linear_model_names[0]][3] == dummy_lags_linear, f"lags don't match for {linear_model_names[0]}"
    assert linear_models_loaded[linear_model_names[1]][0].get_params() == dummy_linear_model_hourly.get_params(), f"linear model doesn't match for {linear_model_names[1]}"
    assert all(linear_models_loaded[linear_model_names[1]][0].predict(X_test_hourly_linear) == hourly_preds_linear), f"linear model predictions do not match for {linear_model_names[1]}"  
    assert len(linear_models_loaded[linear_model_names[1]][1].out_of_sample(10).columns.tolist()) == len(expected_hourly_fourier + expected_order_terms), f"dp doesn't have correct number of columns for {linear_model_names[1]}"
    assert set(linear_models_loaded[linear_model_names[1]][1].out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier + expected_order_terms), f"dp doesn't match columns for {linear_model_names[1]}" 
    assert linear_models_loaded[linear_model_names[1]][2] == None, f"hybrid doesn't match for {linear_model_names[1]}"
    assert linear_models_loaded[linear_model_names[1]][3] == dummy_lags_linear_2, f"lags don't match for {linear_model_names[1]}"
    assert set(non_linear_models_loaded[non_linear_model_names[0]][0].get_params()) == set(dummy_non_linear_model_daily.get_params()), f"non linear model params do not match for {non_linear_model_names[0]}"
    assert all(non_linear_models_loaded[non_linear_model_names[0]][0].predict(X_test_daily_non_linear) == daily_preds_non_linear), f"non linear model does not match expected predictions for {non_linear_model_names[0]}"   
    assert len(non_linear_models_loaded[non_linear_model_names[0]][1].out_of_sample(10).columns.tolist()) == len(expected_daily_fourier), f"dp doesn't have correct number of columns for {non_linear_model_names[0]}"
    assert set(non_linear_models_loaded[non_linear_model_names[0]][1].out_of_sample(10).columns.tolist()) == set(expected_daily_fourier), f"dp doesn't match columns for {non_linear_model_names[0]}" 
    assert non_linear_models_loaded[non_linear_model_names[0]][2] == dummy_non_linear_hybrid, f"hybrid doesn't match for {non_linear_model_names[0]}"
    assert non_linear_models_loaded[non_linear_model_names[0]][3] == dummy_lags_non_linear, f"lags don't match for {non_linear_model_names[0]}"
    assert set(non_linear_models_loaded[non_linear_model_names[1]][0].get_params()) == set(dummy_non_linear_model_hourly.get_params()), f"non linear model params do not match for {non_linear_model_names[1]}"
    assert all(non_linear_models_loaded[non_linear_model_names[1]][0].predict(X_test_hourly_non_linear) == hourly_preds_non_linear), f"non linear model does not match expected predictions for {non_linear_model_names[1]}"  
    assert len(non_linear_models_loaded[non_linear_model_names[1]][1].out_of_sample(10).columns.tolist()) == len(expected_hourly_fourier), f"dp doesn't have correct number of columns for {non_linear_model_names[1]}"
    assert set(non_linear_models_loaded[non_linear_model_names[1]][1].out_of_sample(10).columns.tolist()) == set(expected_hourly_fourier), f"dp doesn't match columns for {non_linear_model_names[1]}"
    assert non_linear_models_loaded[non_linear_model_names[1]][2] == dummy_non_linear_hybrid, f"hybrid doesn't match for {non_linear_model_names[1]}"
    assert non_linear_models_loaded[non_linear_model_names[1]][3] == dummy_lags_non_linear_2, f"lags don't match for {non_linear_model_names[1]}" 

def test_load_design():
    """ test for load_design from training_helpers.py
    """     
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    from test_modelling_helpers import create_ts
    import pandas as pd
    from pandas.testing import assert_series_equal, assert_frame_equal
    import numpy as np
    from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

    # Load config
    config, PROJECT_ROOT =  load_config()

    # We will use a daily time series for the first design and an hourly one for the second
    ts_daily = create_ts("D")
    ts_hourly = create_ts("h")

    # We need to remove the time zone so we can add fourier features
    ts_daily.index = ts_daily.index.tz_localize(None)
    ts_hourly.index = ts_hourly.index.tz_localize(None)

    # We also need to reset the frequencies
    ts_daily = ts_daily.asfreq("D").fillna(0)
    ts_hourly = ts_hourly.asfreq("h").fillna(0)

    # Create dummy designs, lags and dp's, the ts above are the targets
    np.random.seed(37)

    # Fourier features
    daily_fourier = [CalendarFourier(freq = "YE", order = 10), CalendarFourier(freq = "W", order = 5)]
    hourly_fourier = [CalendarFourier(freq = "W", order = 5), CalendarFourier(freq = "D", order = 5)]

    n_daily = len(ts_daily) # Number of rows
    n_hourly = len(ts_hourly) # Number of rows
    dp_daily = DeterministicProcess(index = ts_daily.index, order = 3, seasonal = False, constant = True, additional_terms = daily_fourier)
    dp_daily_2 = DeterministicProcess(index = ts_daily.index, order = 0, seasonal=False, constant = True, additional_terms = daily_fourier) 
    dp_hourly = DeterministicProcess(index = ts_hourly.index, order = 3, seasonal = False, constant = True, additional_terms = hourly_fourier)
    dp_hourly_2 = DeterministicProcess(index = ts_hourly.index, order = 0, seasonal = False, constant = False, additional_terms = hourly_fourier)
    

    dummy_X_daily_linear = dp_daily.in_sample()
    dummy_X_daily_non_linear = dp_daily_2.in_sample()
    dummy_X_hourly_linear = dp_hourly.in_sample()
    dummy_X_hourly_non_linear = dp_hourly_2.in_sample()

    lags_daily = [1,2,3,4,5,6]
    lags_hourly = [1, 2, 3, 480]
    
    linear_design_names = ["test_linear_design", "test_linear_design_2"]
    non_linear_design_names = ["test_non_linear_design", "test_non_linear_design_2"]


    linear_design = {
        linear_design_names[0]: (dummy_X_daily_linear, ts_daily, dp_daily, lags_daily), 
        linear_design_names[1]: (dummy_X_hourly_linear, ts_hourly, dp_hourly, lags_hourly) 
    }

    non_linear_design = {
        non_linear_design_names[0]: (dummy_X_daily_non_linear, ts_daily, dp_daily_2, lags_daily),
        non_linear_design_names[1]: (dummy_X_hourly_non_linear, ts_daily, dp_hourly_2, lags_hourly)
    }

    # Save the models
    sig = "test_model_sig"
    training_helpers.save_design(linear_design, non_linear_design, sig)

    # Load the models
    linear_design_loaded, non_linear_design_loaded = training_helpers.load_design(sig)

    # Expected fourier terms
    expected_daily_fourier = [f"cos({x},freq=YE-DEC)" for x in range(1, 11)] + [f"sin({x},freq=YE-DEC)" for x in range(1, 11)] + \
                            [f"cos({x},freq=W-SUN)" for x in range(1, 6)] + [f"sin({x},freq=W-SUN)" for x in range(1, 6)]
    expected_hourly_fourier = [f"cos({x},freq=D)" for x in range(1, 6)] + [f"sin({x},freq=D)" for x in range(1, 6)] + \
                            [f"cos({x},freq=W-SUN)" for x in range(1, 6)] + [f"sin({x},freq=W-SUN)" for x in range(1, 6)]

    # Expected order terms
    expected_order_terms = ["const", "trend", "trend_squared", "trend_cubed"]

    # Checks
    assert len(linear_design_loaded.keys()) == len(linear_design_names), "linear design loaded has incorrect key length"
    assert set(linear_design_loaded.keys()) == set(linear_design_names), "linear design loaded has incorrect keys"
    assert len(non_linear_design_loaded.keys()) == len(non_linear_design_names), "non linear design loaded has incorrect key length"
    assert set(non_linear_design_loaded.keys()) == set(non_linear_design_names), "non linear design loaded has incorrect keys"
    assert_frame_equal(
        linear_design_loaded[linear_design_names[0]][0],
        dummy_X_daily_linear,
        check_dtype = True,
        check_exact = True, 
    ), f"X doesn't match for {linear_design_names[0]}"
    assert_series_equal(
        linear_design_loaded[linear_design_names[0]][1],
        ts_daily,
        check_dtype = True,
        check_freq = True,
        check_exact = True,
        check_names = True,
    ), f"ts doesn't match for {linear_design_names[0]}"
    assert len(linear_design_loaded[linear_design_names[0]][2].out_of_sample(10).columns.to_list()) == len(expected_daily_fourier + expected_order_terms), f"dp doesn't have correct number of columns for {linear_design_names[0]}"
    assert set(linear_design_loaded[linear_design_names[0]][2].out_of_sample(10).columns.to_list()) == set(expected_daily_fourier + expected_order_terms), f"dp doesn't have correct columns for {linear_design_names[0]}"
    assert linear_design_loaded[linear_design_names[0]][3] == lags_daily, f"lags don't match for {linear_design_names[0]}"
    assert_frame_equal(
        linear_design_loaded[linear_design_names[1]][0],
        dummy_X_hourly_linear,
        check_dtype = True,
        check_exact = True, 
    ), f"X doesn't match for {linear_design_names[1]}"
    assert_series_equal(
        linear_design_loaded[linear_design_names[1]][1],
        ts_hourly,
        check_dtype = True,
        check_freq = True,
        check_exact = True,
        check_names = True,
    ), f"ts doesn't match for {linear_design_names[1]}"
    assert len(linear_design_loaded[linear_design_names[1]][2].out_of_sample(10).columns.to_list()) == len(expected_hourly_fourier + expected_order_terms), f"dp doesn't have correct number of columns for {linear_design_names[1]}"
    assert set(linear_design_loaded[linear_design_names[1]][2].out_of_sample(10).columns.to_list()) == set(expected_hourly_fourier + expected_order_terms), f"dp doesn't have correct columns for {linear_design_names[1]}"
    assert linear_design_loaded[linear_design_names[1]][3] == lags_hourly, f"lags don't match for {linear_design_names[1]}"

    assert_frame_equal(
        non_linear_design_loaded[non_linear_design_names[0]][0],
        dummy_X_daily_non_linear,
        check_dtype = True,
        check_exact = True, 
    ), f"X doesn't match for {non_linear_design_names[0]}"
    assert_series_equal(
        non_linear_design_loaded[non_linear_design_names[0]][1],
        ts_daily,
        check_dtype = True,
        check_freq = True,
        check_exact = True,
        check_names = True,
    ), f"ts doesn't match for {non_linear_design_names[0]}"
    # dp_daily_2 has order=0, constant=True, so only "const" plus fourier terms
    expected_daily_non_linear_terms = ["const"] + expected_daily_fourier
    assert len(non_linear_design_loaded[non_linear_design_names[0]][2].out_of_sample(10).columns.to_list()) == len(expected_daily_non_linear_terms), f"dp doesn't have correct number of columns for {non_linear_design_names[0]}"
    assert set(non_linear_design_loaded[non_linear_design_names[0]][2].out_of_sample(10).columns.to_list()) == set(expected_daily_non_linear_terms), f"dp doesn't have correct columns for {non_linear_design_names[0]}"
    assert non_linear_design_loaded[non_linear_design_names[0]][3] == lags_daily, f"lags don't match for {non_linear_design_names[0]}"

    assert_frame_equal(
        non_linear_design_loaded[non_linear_design_names[1]][0],
        dummy_X_hourly_non_linear,
        check_dtype = True,
        check_exact = True, 
    ), f"X doesn't match for {non_linear_design_names[1]}"
    assert_series_equal(
        non_linear_design_loaded[non_linear_design_names[1]][1],
        ts_daily,
        check_dtype = True,
        check_freq = True,
        check_exact = True,
        check_names = True,
    ), f"ts doesn't match for {non_linear_design_names[1]}"
    # dp_hourly_2 has order=0, constant=False, so only fourier terms
    expected_hourly_non_linear_terms = expected_hourly_fourier
    assert len(non_linear_design_loaded[non_linear_design_names[1]][2].out_of_sample(10).columns.to_list()) == len(expected_hourly_non_linear_terms), f"dp doesn't have correct number of columns for {non_linear_design_names[1]}"
    assert set(non_linear_design_loaded[non_linear_design_names[1]][2].out_of_sample(10).columns.to_list()) == set(expected_hourly_non_linear_terms), f"dp doesn't have correct columns for {non_linear_design_names[1]}"
    assert non_linear_design_loaded[non_linear_design_names[1]][3] == lags_hourly, f"lags don't match for {non_linear_design_names[1]}"

def test_save_lags_and_load_lags():
    """Test save_lags and load_lags functions in training_helpers.py"""
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config

    # Load config
    config, PROJECT_ROOT = load_config()

    # Constants
    LAGS_PREFFIX = config["saving"]["lags_preffix"]
    SAVED_OBJECTS_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["saved_objects_path"]
 
    sig = f"testsig_lags"
    lags = [1, 2, 3, 7, 14]
    series_type = "daily"

    # Save lags
    training_helpers.save_lags(lags, series_type, sig)

    # Check that the file exists
    f = SAVED_OBJECTS_DIR / f"{LAGS_PREFFIX}_{series_type}_{sig}.pkl"
    assert f.exists(), "lags file does not exist"

    # Load lags
    loaded_lags = training_helpers.load_lags(series_type, sig)
    assert loaded_lags == lags, "Loaded lags do not match saved lags"

def test_save_hyperparams_and_load_hyperparams():
    """Test save_hyperparams and load_hyperparams functions in training_helpers.py"""
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    import random

    # Load config 
    config, PROJECT_ROOT = load_config()
    
    # Constants
    HYPERPARAMS_PREFFIX = config["saving"]["hyperparams_preffix"]
    SAVED_OBJECTS_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["saved_objects_path"]

    sig = f"testsig_hyperparams"
    hyperparams = {"alpha": 0.1, "beta": 0.5, "max_depth": 3}

    # Save hyperparams
    training_helpers.save_hyperparams(hyperparams, sig)

    # Check that the file exists
    f = SAVED_OBJECTS_DIR / f"{HYPERPARAMS_PREFFIX}_{sig}.pkl"
    assert f.exists(), "hyperparam file does not exist"

    # Load hyperparams
    loaded_hyperparams = training_helpers.load_hyperparams(sig)
    assert loaded_hyperparams == hyperparams, "Loaded hyperparams do not match saved hyperparams"

def test_save_obj_and_load_obj():
    """Test save_obj and load_obj functions in training_helpers.py"""
    from jfk_taxis import training_helpers
    from jfk_taxis import load_config
    
    # Load config
    config, PROJECT_ROOT = load_config()
    
    # Constants
    SAVED_OBJECTS_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["saved_objects_path"]
 
    sig = f"testsig_obj"
    # Use a simple object, e.g., a dict
    obj = {"foo": [1, 2, 3], "bar": {"baz": 42}}

    # Save object
    training_helpers.save_obj(obj, sig)

    # Check that the file exists
    f =  SAVED_OBJECTS_DIR / f"{sig}.pkl"
    assert f.exists(), "obj file does not exist"

    # Load object
    loaded_obj = training_helpers.load_obj(sig)
    assert loaded_obj == obj, "Loaded object does not match saved object"
