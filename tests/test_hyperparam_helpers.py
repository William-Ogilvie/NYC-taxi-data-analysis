"""
test_hyperparam_helpers.py
============================

Unit tests for hyperparam_helpers.py
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
    assert daily_non_linear_obj == avg_daily_non_linear_mae
    assert daily_hybrid_obj == avg_daily_hybrid_mae

    # Same for hourly
    hybrid_linear_model = LinearRegression(fit_intercept=False)
    hourly_non_linear_obj = hyperparam_helpers.objective_optuna(trial, hourly_fold_dict, hourly_steps, None, offset_list)
    hourly_hybrid_obj = hyperparam_helpers.objective_optuna(trial, hourly_fold_dict, hourly_steps, hybrid_linear_model, offset_list)

    # Now we compare the results
    assert hourly_non_linear_obj == avg_hourly_non_linear_mae
    assert hourly_hybrid_obj == avg_hourly_hybrid_mae




