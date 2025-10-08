"""
test_hyperparam_helpers.py
============================

Unit tests for hyperparam_helpers.py. Again we won't test test_hyperparams because it essentially calls all the other smaller functions and plots the output. So just checking the plots manually is enough.
"""

from jfk_taxis import hyperparam_helpers


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
        assert len(y_test) == daily_test_size, "y_test length incorrect"

        # y_test 1 time step after y_train
        assert y_test.index[0].tz_localize(None) == y_train.index[-1] +  np.timedelta64(1, "D"), "y_test does not start 1 time step after y_train"

        # y_test formatted correctly (this isn't given as y_test doesn't come from preprocesss so worth checking)
        for i in range(0, y_test.shape[0]-1):
            assert y_test.index[i+1] == y_test.index[i] + np.timedelta64(1, "D"), "y_test not formatted correctly"

    for key, value in hourly_fold_dict.items():
        y_train = value[1]
        y_test = value[4]

        # Length of y_test
        assert len(y_test) == hourly_test_size, "y_test length incorrect"

        # y_test 1 time step after y_train
        assert y_test.index[0].tz_localize(None) == y_train.index[-1] +  np.timedelta64(1, "h"), "y_test does not start 1 time step after y_train"

        # y_test formatted correctly (this isn't given as y_test doesn't come from preprocesss so worth checking)
        for i in range(0, y_test.shape[0]-1):
            assert y_test.index[i+1] == y_test.index[i] + np.timedelta64(1, "h"), "y_test not formatted correctly"

def test_objective_optuna():
    """ test for the objective_optuna function in hyperparam_helpers.py.
    """
    import pandas as pd
    import numpy as np
    import cupy as cp
    import optuna
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_absolute_error
    import copy
    from jfk_taxis import hyperparam_helpers
    from jfk_taxis import forecast_helpers
    from jfk_taxis import load_config


    # Load config
    config, PROJECT_ROOT = load_config()

    # We will do two sets of sample data one with daily data and one with hourly data 
    series_daily_full = pd.Series(data = np.random.uniform(3500, 5500, size = 365*3), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3, freq = "D"))
    series_hourly_full = pd.Series(data = np.random.uniform(50, 400, size = 365*3*24), index = pd.date_range(start = "2021-01-01 00:00:00+0000", periods = 365*3*24, freq = "h"))

    n_splits = 5

    daily_test_size = 60
    hourly_test_size = 24*7*2

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

    # We will use a fixed optuna trial object to ensure consistent results
    trial = optuna.trial.FixedTrial({
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 1.0,
        "gamma": 0.1,
    })

    offset_list = [0, 7, 14]
    daily_steps = 30
    hourly_steps = 168

    # We first calculate the average MAE across all folds and offsets ourselves, we will do this for a linear, non linear and hybrid model
    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=1.0,
        gamma=0.1,
        random_state=config["xgboost_setup"]["random_state"],
        eval_metric=config["xgboost_setup"]["eval_metric"],
        tree_method=config["xgboost_setup"]["tree_method"],
        device=config["xgboost_setup"]["device"]
    )

    daily_non_linear_mae_list = []
    daily_hybrid_mae_list = []
    for value in daily_fold_dict.values():
        X_train = value[0]
        y_train = value[1]
        dp = value[2]
        lags = value[3] 
        y_test = value[4]

        # Non linear model
        non_linear_model = copy.deepcopy(xgb_model)

        # Hybrid non linear model
        hybrid_model = copy.deepcopy(xgb_model)

        # Linear model
        linear_model = LinearRegression(fit_intercept=False)

        # To pass for linear and non linear 
        hybrid = None

        # Set gpu
        if config["xgboost_setup"]["device"] == "cuda":
            gpu = True
        else: 
            gpu = False

        # Fit the models
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()
        linear_model.fit(X_train_np, y_train_np)
        y_resid = y_train_np - linear_model.predict(X_train_np)

        # If usign gpu we need to fit as cupy arrays
        if gpu:
            X_train_cp = cp.asarray(X_train_np)
            y_train_cp = cp.asarray(y_train_np)
            non_linear_model.fit(X_train_cp, y_train_cp)
            y_resid_cp = cp.asarray(y_resid)
            hybrid_model.fit(X_train_cp, y_resid_cp)
        else:
            non_linear_model.fit(X_train_np, y_train_np)
            hybrid_model.fit(X_train_np, y_resid)

        tmp_mae_non_linear = []
        tmp_mae_hybrid = []
        for offset in offset_list:
            y_preds_non_linear = forecast_helpers.forecast(non_linear_model, y_train, lags, daily_steps, offset, dp, hybrid, gpu)

            y_preds_hybrid = forecast_helpers.forecast(linear_model, y_train, lags, daily_steps, offset, dp, hybrid_model, False)

            y_test_offset = y_test.iloc[offset: offset+daily_steps]


            mae_non_linear = mean_absolute_error(y_test_offset, y_preds_non_linear)
            mae_hybrid = mean_absolute_error(y_test_offset, y_preds_hybrid)

            tmp_mae_non_linear.append(mae_non_linear)
            tmp_mae_hybrid.append(mae_hybrid)

        # Average mae for this fold
        daily_non_linear_mae_list.append(np.mean(tmp_mae_non_linear))
        daily_hybrid_mae_list.append(np.mean(tmp_mae_hybrid))

    avg_daily_non_linear_mae = sum(daily_non_linear_mae_list) / len(daily_non_linear_mae_list)
    avg_daily_hybrid_mae = sum(daily_hybrid_mae_list) / len(daily_hybrid_mae_list)

    hourly_non_linear_mae_list = []
    hourly_hybrid_mae_list = []
    for value in hourly_fold_dict.values():
        X_train = value[0]
        y_train = value[1]
        dp = value[2]
        lags = value[3] 
        y_test = value[4]

        # Non linear model
        non_linear_model = copy.deepcopy(xgb_model)

        # Hybrid non linear model
        hybrid_model = copy.deepcopy(xgb_model)

        # Linear model
        linear_model = LinearRegression(fit_intercept=False)

        # To pass for linear and non linear 
        hybrid = None

        # Set gpu
        if config["xgboost_setup"]["device"] == "cuda":
            gpu = True
        else: 
            gpu = False

        # Fit the models
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()
        linear_model.fit(X_train_np, y_train_np)
        y_resid = y_train_np - linear_model.predict(X_train_np)

        # If usign gpu we need to fit as cupy arrays
        if gpu:
            X_train_cp = cp.asarray(X_train_np)
            y_train_cp = cp.asarray(y_train_np)
            non_linear_model.fit(X_train_cp, y_train_cp)
            y_resid_cp = cp.asarray(y_resid)
            hybrid_model.fit(X_train_cp, y_resid_cp)
        else:
            non_linear_model.fit(X_train_np, y_train_np)
            hybrid_model.fit(X_train_np, y_resid)

        tmp_mae_non_linear = []
        tmp_mae_hybrid = []
        for offset in offset_list:
            y_preds_non_linear = forecast_helpers.forecast(non_linear_model, y_train, lags, hourly_steps, offset, dp, hybrid, gpu)

            y_preds_hybrid = forecast_helpers.forecast(linear_model, y_train, lags, hourly_steps, offset, dp, hybrid_model, False)

            y_test_offset = y_test.iloc[offset: offset+hourly_steps]

            mae_non_linear = mean_absolute_error(y_test_offset, y_preds_non_linear)
            mae_hybrid = mean_absolute_error(y_test_offset, y_preds_hybrid)

            tmp_mae_non_linear.append(mae_non_linear)
            tmp_mae_hybrid.append(mae_hybrid)

        # Average mae for this fold
        hourly_non_linear_mae_list.append(np.mean(tmp_mae_non_linear))
        hourly_hybrid_mae_list.append(np.mean(tmp_mae_hybrid))

    avg_hourly_non_linear_mae = np.mean(hourly_non_linear_mae_list)
    avg_hourly_hybrid_mae = np.mean(hourly_hybrid_mae_list)

    # Now we use the objective function and test we get the same results
    hybrid_linear_model = LinearRegression(fit_intercept=False)
    daily_non_linear_obj = hyperparam_helpers.objective_optuna(trial, daily_fold_dict, daily_steps, None, offset_list)
    daily_hybrid_obj = hyperparam_helpers.objective_optuna(trial, daily_fold_dict, daily_steps, hybrid_linear_model, offset_list)

    # Now we compare the results
    assert daily_non_linear_obj == avg_daily_non_linear_mae, "mae of daily non linear model does not match"
    assert daily_hybrid_obj == avg_daily_hybrid_mae, "mae of daily hybrid model does not match"

    # Same for hourly
    hybrid_linear_model = LinearRegression(fit_intercept=False)
    hourly_non_linear_obj = hyperparam_helpers.objective_optuna(trial, hourly_fold_dict, hourly_steps, None, offset_list)
    hourly_hybrid_obj = hyperparam_helpers.objective_optuna(trial, hourly_fold_dict, hourly_steps, hybrid_linear_model, offset_list)

    # Now we compare the results
    assert hourly_non_linear_obj == avg_hourly_non_linear_mae, "mae of hourly non linear model does not match"
    assert hourly_hybrid_obj == avg_hourly_hybrid_mae, "mae of hourly hybrid model does not match"

def test_split_params():
    """ test the split_params function in hyperparam_helpers.py
    """    
    # Create sample params
    params = {
        "daily_hybrid_order1": {"param_2": 0.1, "param_3": 5}, 
        "reduced_daily_base_non_linear": {"param_5": 10, "param_6": 0.01},
        "reduced_hourly_hybrid_order1": {"param_2": 0.2, "param_3": 10}, 
        "hourly_base_non_linear": {"param_5": 20, "param_6": 0.02},
    }

    daily_non_linear_dict, daily_hybrid_dict, hourly_non_linear_dict, hourly_hybrid_dict = hyperparam_helpers.split_params(params)

    # Check the contents of each dictionary
    assert daily_non_linear_dict.keys() == {"reduced_daily_base_non_linear"}, "keys of daily non linear dict incorrect"
    assert daily_non_linear_dict["reduced_daily_base_non_linear"] == {"param_5": 10, "param_6": 0.01}, "params of daily non linear dict incorrect"
    assert daily_hybrid_dict.keys() == {"daily_hybrid_order1"}, "keys of daily hybrid dict incorrect"
    assert daily_hybrid_dict["daily_hybrid_order1"] == {"param_2": 0.1, "param_3": 5}, "params of daily hybrid dict incorrect"
    assert hourly_non_linear_dict.keys() == {"hourly_base_non_linear"}, "keys of hourly non linear dict incorrect"
    assert hourly_non_linear_dict["hourly_base_non_linear"] == {"param_5": 20, "param_6": 0.02}, "params of hourly non linear dict incorrect"
    assert hourly_hybrid_dict.keys() == {"reduced_hourly_hybrid_order1"}, "keys of hourly hybrid dict incorrect"
    assert hourly_hybrid_dict["reduced_hourly_hybrid_order1"] == {"param_2": 0.2, "param_3": 10}, "params of hourly hybrid dict incorrect"


def test_define_model():
    """ test for define_model function in hyperparam_helpers.py
    """
    import optuna
    from xgboost import XGBRegressor
    from jfk_taxis import hyperparam_helpers
    from jfk_taxis import load_config

    # Load config
    config, PROJECT_ROOT = load_config()

    # Create a fixed trial with specific hyperparameters
    trial = optuna.trial.FixedTrial({
        "n_estimators": 250,
        "learning_rate": 0.07,
        "max_depth": 4,
        "min_child_weight": 2.5,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.5,
        "gamma": 0.2,
    })

    # Call define_model
    model = hyperparam_helpers.define_model(trial)

    # Check that the returned object is an XGBRegressor
    assert isinstance(model, XGBRegressor), "Model should be an XGBRegressor"

    # Check that hyperparameters were set correctly
    assert model.n_estimators == 250, "n_estimators should be 250"
    assert model.learning_rate == 0.07, "learning_rate should be 0.07"
    assert model.max_depth == 4, "max_depth should be 4"
    assert model.min_child_weight == 2.5, "min_child_weight should be 2.5"
    assert model.subsample == 0.75, "subsample should be 0.75"
    assert model.colsample_bytree == 0.85, "colsample_bytree should be 0.85"
    assert model.reg_lambda == 5.0, "reg_lambda should be 5.0"
    assert model.reg_alpha == 0.5, "reg_alpha should be 0.5"
    assert model.gamma == 0.2, "gamma should be 0.2"

    # Check that config parameters were set correctly
    assert model.random_state == config["xgboost_setup"]["random_state"], "random_state should match config"
    assert model.eval_metric == config["xgboost_setup"]["eval_metric"], "eval_metric should match config"
    assert model.tree_method == config["xgboost_setup"]["tree_method"], "tree_method should match config"
    assert model.device == config["xgboost_setup"]["device"], "device should match config"


def test_test_hyperparams():
    """ test for test_hyperparams function in hyperparam_helpers.py
    
    This is a smoke test - we just verify the function runs without errors with sample data. 
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from jfk_taxis import hyperparam_helpers
    from jfk_taxis import load_config

    # Load config
    config, PROJECT_ROOT = load_config()

    # Use non-interactive backend and stub show
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Create sample time series data (timezone-naive to avoid matplotlib issues)
        # Then convert to UTC which is what the actual functions expect
        ts_daily_train = pd.Series(
            data=np.random.uniform(3500, 5500, size=365*2), 
            index=pd.date_range(start="2021-01-01", periods=365*2, freq="D", tz="UTC")
        )
        ts_daily_test = pd.Series(
            data=np.random.uniform(3500, 5500, size=60), 
            index=pd.date_range(start="2023-01-01", periods=60, freq="D", tz="UTC")
        )
        
        ts_hourly_train = pd.Series(
            data=np.random.uniform(50, 400, size=365*24*2), 
            index=pd.date_range(start="2021-01-01", periods=365*24*2, freq="h", tz="UTC")
        )
        ts_hourly_test = pd.Series(
            data=np.random.uniform(50, 400, size=168*2), 
            index=pd.date_range(start="2023-01-01", periods=168*2, freq="h", tz="UTC")
        )

        # Create sample hyperparameter dictionaries (minimal for smoke test)
        dict_full = {
            config["hyperparameter_tuning"]["daily_linear_key"]: {
                "test_daily_non_linear": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "min_child_weight": 1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "reg_alpha": 1.0,
                    "gamma": 0.1,
                    "random_state": config["xgboost_setup"]["random_state"],
                    "eval_metric": config["xgboost_setup"]["eval_metric"],
                    "tree_method": config["xgboost_setup"]["tree_method"],
                    "device": config["xgboost_setup"]["device"]
                }
            }
        }

        # Define steps and offsets (small values for quick testing)
        daily_steps = [7]
        hourly_steps = [24]
        daily_offsets = [0, 7]
        hourly_offsets = [0, 24]
        daily_offsets_to_show = [0]
        hourly_offsets_to_show = [0]

        # Run the function - should not raise any errors
        hyperparam_helpers.test_hyperparams(
            dict_full, 
            ts_daily_train, 
            ts_daily_test, 
            ts_hourly_train, 
            ts_hourly_test,
            daily_steps, 
            hourly_steps, 
            daily_offsets, 
            hourly_offsets, 
            daily_offsets_to_show, 
            hourly_offsets_to_show
        )

        # Close plots to avoid warnings
        plt.close("all")

    finally:
        plt.show = old_show












