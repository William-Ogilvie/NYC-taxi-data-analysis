"""
hyperparam_helpers
=================

This module contains helper functions for hyperparameter tuning using hyperopt. It includes both functions to make the validation folds and the objective function itself.
Along with a wrapper function to allow passing additional parameters to the objective function when using fmin from hyperopt.
"""

# --- Imports ---
from hyperopt import STATUS_OK
import xgboost as xgb
from .forecast_helpers import forecast, run_forecasts, preprocess, add_lags_to_dict, run_forecasts_diff_lags
from .loading_helpers import load_config
from .training_helpers import load_models, load_design, load_process_lags
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import time
import cupy as cp
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import copy

# --- Load config ---
config, PROJECT_ROOT = load_config()

# --- Constants ---
LINEAR_MODEL_PREFIX = config["model_naming"]["linear_model_prefix"]
REDUCED_DAILY_DEFAULT_NON_LINEAR = config["model_naming"]["reduced_daily_non_linear_model_prefix"]
REDUCED_HYBRID_DAILY_MODEL_PREFIX = config["model_naming"]["reduced_daily_hybrid_model_prefix"]
REDUCED_HOURLY_DEFAULT_NON_LINEAR = config["model_naming"]["reduced_hourly_non_linear_model_prefix"]
REDUCED_HYBRID_HOURLY_MODEL_PREFIX = config["model_naming"]["reduced_hourly_hybrid_model_prefix"]


# --- Functions ---
def create_val_data(n_splits: int, test_size: int, lags: list[int], constant: bool, order: int, fourier_features: list[str], time_step: str, ts: pd.Series) -> dict:
    """ Creates the folds to be used in cross validation for hyperparameter tuning

    Args:
        n_splits (int): number of splits/folds
        test_size (int): size of the test set for each fold
        lags (list[int]): list of lags to use
        constant (bool): whether the deterministic process should have a constant
        order (int): order of the linear trend in the deterministic process
        fourier_features (list[str]): list of fourier features to use
        time_step (str): time step of the time series (e.g. "h", "D")
        ts (pd.Series): time series data

    Returns:
        dict: dictionary containing the folds with keys as fold_0, fold_1, ..., each value is a tuple (X_train, y_train, dp, y_test). dp is the deterministic process fitted on the training data and will be used to create the out of sample features for forecasting
    """        
    
    # We will store all of the folds in a dict
    fold_dict = {}
    
    # We need to split the ts using TimeSeriesSplit 
    tscv= TimeSeriesSplit(n_splits = n_splits, test_size = test_size)
    
    for fold, (train_index, test_index) in enumerate(tscv.split(ts)):
        print(f"Fold {fold}")
        print(train_index)
        # We need to preprocess the training portion of the fold
        ts_train = ts.iloc[train_index].copy()
        (X_train, y_train, dp) = preprocess(lags, constant, order, fourier_features, time_step, ts_train)

        # We don't need to preprocess the test portion of the fold because we are going to pass the deterministic process and use dp.out_sample()
        # when forecasting as we are doing a multistep forecast and need to build lags as we go. 
        y_test = ts.iloc[test_index].copy()


        # To improve memory usage set to float32
        X_train = X_train.astype("float32")
        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")
        
        fold_dict[f"fold_{fold}"] = (X_train, y_train, dp, y_test)

    return fold_dict


def objective(space: dict, fold_dict: dict, lags: list[int], steps: int, hybrid: LinearRegression | None) -> dict:
    """ Objective function for hyperparameter tuning using hyperopt

    Args:
        space (dict): hyperparameter space
        fold_dict (dict): dict containing the folds with keys as fold_0, fold_1, ..., each value is a tuple (X_train, y_train, dp, y_test)
        lags (list[int]): list of lags
        steps (list[int]): number of steps to forecast
        hybrid (LinearRegression | None): hybrid model to use, if None then no hybrid model is used

    Returns:
        dict: dictionary containing the computed loss (mean MAE across folds) and the status 
    """     

    model = XGBRegressor(
        n_estimators = space["n_estimators"],
        
        learning_rate = space["learning_rate"],

        max_depth = space["max_depth"],
        min_child_weight = space["min_child_weight"],

        subsample = space["subsample"],
        colsample_bytree = space["colsample_bytree"],

        reg_lambda = space["reg_lambda"],
        reg_alpha = space["reg_alpha"],

        gamma = space["gamma"],

        random_state = space["random_state"],
        #early_stopping_rounds = space["early_stopping_rounds"],
        eval_metric = space["eval_metric"],
        

        # Tree method hist will eseentially bin feature values into histograms and consider 
        # and then only considers splits at bin boundaries. 
        # If you have a gpu it is advised you use it particularly for training the hourly dataset, to do set device = "cuda" (you may need to install the gpu version of xgboost manually with conda install -c conda-forge py-xgboost=*=cuda*)
        tree_method = space["tree_method"],
        device = space["device"] # use gpu
        )

    # If we are in the hybrid case then the model above is actually the "hybrid" part used in forecasting, and we have passed the linear regression as the hybrid component of this function
    # so we need to swap theses two
    if hybrid is not None:
        tmp = model
        model = hybrid
        hybrid = tmp

    maes = []
    for value in fold_dict.values():
        X_train = value[0]
        y_train = value[1]
        dp = value[2]
        y_test = value[3]
 

        # We are going to use early stopping, now to avoid very long computations we will take a slight shorcut in that the validation set will have 
        # the "real lags" rather than the lags computed during the forecast on the test set of the fold. 

        # # Get the number of rows in X_train
        # num_rows = X_train.shape[0]

        # # We will validate on about 10% of the data
        # split_row = int(num_rows * 0.9) # this tells us which row to split on
        # X_val = X_train.iloc[split_row:]
        # y_val = y_train.iloc[split_row:]
    

        # X_train = X_train.iloc[:split_row]
        # y_train = y_train.iloc[:split_row]

        # Convert to numpy arrays before fitting model
        X_train_np = X_train.to_numpy(copy = True)
        y_train_np = y_train.to_numpy(copy = True)


        # X_val_np = X_val.to_numpy(copy = True)
        # y_val_np = y_val.to_numpy(copy = True)

        # Time fitting the model
        start_fit = time.time()

        # If there is no hybrid model we can just fit the model to the data
        if hybrid is None:
            # Convert to CuPy arrays if using GPU
            if config["xgboost_setup"]["device"] == "cuda":
                X_train_cp = cp.asarray(X_train_np)
                y_train_cp = cp.asarray(y_train_np)

                # Fit the model
                model.fit(X_train_cp, y_train_cp)
            else:
                model.fit(X_train_np, y_train_np)

        # If we have a hyrid model then this will need to be fit to the data first (as it is the linear part) and then we pass the XGBoost part to the forecast as hybrid
        else: 
            # Fit model to the data (we fit to the numpy arrays as model is a linear regression)
            model.fit(X_train_np, y_train_np)

            # We now need to compute the residuals and fit the hybrid model to these
            y_fit = model.predict(X_train_np)
            y_resid = y_train_np - y_fit

            # Convert to CuPy arrays if using GPU and fit the model
            if config["xgboost_setup"]["device"] == "cuda":
                X_train_cp = cp.asarray(X_train_np)
                y_resid_cp = cp.asarray(y_resid)
                hybrid.fit(X_train_cp, y_resid_cp)
            else:
                hybrid.fit(X_train_np, y_resid)

        end_fit = time.time()

        # Time forecasting
        start_fore = time.time()

        # We want to use gpu for predictions
        gpu = True 

        # Run the forecast for the required steps
        y_preds = forecast(model, y_train, lags, steps, dp, hybrid, gpu)

        end_fore = time.time()
    
        # Report timings
        print(f"Fit time: {end_fit - start_fit:.2f} seconds")
        print(f"Predict time: {end_fore - start_fore:.4f} seconds")

        # # See iterations stopped at
        # print("Best iteration:", model.best_iteration)
        # print("Best score:", model.best_score)

        # Compute MAE
        mae = mean_absolute_error(y_preds, y_test)
        maes.append(mae) 

    mean_mae = sum(maes) / len(maes)
    print("MAEs:", maes)
    print("Avg MAE:", mean_mae)
    return {'loss': mean_mae, 'status': STATUS_OK}


def wrapped_objective(space: dict) -> dict:
    """ Wrapper function for the objective to include additional parameters.

    Args:
        space (dict): hyperparameter space

    Returns:
        dict: same return as objective function, dict of loss and status
    """    
    return objective(space, wrapped_objective.fold_dict, wrapped_objective.lags, wrapped_objective.steps, wrapped_objective.hybrid)

def split_params(params: dict) -> tuple[dict, dict, dict, dict]:
    """ Splits the hyperparam dicts into four dicts, daily_non_linear, daily_hybrid, hourly_non_linear, hourly_hybrid

    Args:
        params (dict): dict of hyperparameters, keys are the signatures of the model.

    Returns:
        tuple[dict, dict, dict, dict]: the four specifided dicts from above
    """    
    daily_non_linear_dict = {}
    daily_hybrid_dict = {}
    hourly_non_linear_dict = {}
    hourly_hybrid_dict = {}

    # Split into the four dicts
    for key, value in params.items():
        if ("daily" in key) and ("hybrid" in key):
            daily_hybrid_dict[key] = value
        elif "daily" in key:
            daily_non_linear_dict[key] = value
        elif ("hourly" in key) and ("hybrid" in key):
            hourly_hybrid_dict[key] = value
        elif "hourly" in key:
            hourly_non_linear_dict[key] = value

    return daily_non_linear_dict, daily_hybrid_dict, hourly_non_linear_dict, hourly_hybrid_dict


def test_hyperparams(dict_full: dict, daily_lags: list[int], used_hourly_lags: list[int], ts_daily_train: pd.Series, ts_daily_test: pd.Series, ts_hourly_train: pd.Series, ts_hourly_test: pd.Series, daily_steps: list[int], hourly_steps: list[int]) -> None:
    """ Tests the hyperparameters by loading the previous models and design matrices, defining a new model with the hyperparameters and fitting it to the training data.
        Outputs the same forecasts using run_forecasts as in the modelling notebook.

    Args:
        dict_full (dict): dict of dict of hyperparamters, split into four keys, "daily", "daily_hybrid", "hourly", "hourly_hybrid", then each dict has values of the hyperparameters key of the model signature
        daily_lags (list[int]): list of daily lags to use for the model
        used_hourly_lags (list[int]): list of hourly lags to use for the model
        ts_daily_train (pd.Series): training data for the daily model
        ts_daily_test (pd.Series): test data for the daily model
        ts_hourly_train (pd.Series): training data for the hourly model
        ts_hourly_test (pd.Series): test data for the hourly model
        daily_steps (list[int]): list of steps to forecast for the daily model
        hourly_steps (list[int]): list of steps to forecast for the hourly model
    """    

    # We will need the lags for the older models
    old_daily_lags, old_hourly_lags = load_process_lags()



    for key, value in dict_full.items():
        for key2, value2 in value.items():
            if key == config["hyperparameter_tuning"]["daily_linear_key"]:
                # Load the previous non linear model
                linear_models_loaded, non_linear_models_loaded = load_models(config["model_sigs"]["daily_non_linear_reduced"])

                # Load the previous non linear design matrix
                linear_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["daily_non_linear_reduced"])

                # Get design, target and dp
                X = non_linear_design_loaded[REDUCED_DAILY_DEFAULT_NON_LINEAR][0]
                y = non_linear_design_loaded[REDUCED_DAILY_DEFAULT_NON_LINEAR][1]
                dp = non_linear_design_loaded[REDUCED_DAILY_DEFAULT_NON_LINEAR][2]

                # Define the model, **value2 will pass the dict of hyperparamters to the XGBRegressor 
                new_non_linear = xgb.XGBRegressor(
                    **value2 
                )

                # Convert to CuPy arrays before fitting model
                X_cp = cp.asarray(X)
                y_cp = cp.asarray(y)

                # Fit the model
                new_non_linear.fit(X_cp,y_cp)

                # Add this non linear model to the dict of non linear models
                non_linear_models_loaded[key2] = (new_non_linear, dp, None)


                # Because we have two sets of lags we need to use run_forecasts_diff_lags, which means we have to add the lags to the loaded dicts

                # Now we want to load the full linear models (not the reduced ones)
                full_linear_models_loaded, full_non_linear_models_loaded = load_models(config["model_sigs"]["daily_linear"])    

                # Add lags to full linear models 
                full_linear_models_loaded = add_lags_to_dict(full_linear_models_loaded, old_daily_lags)

                # Add lags to reduced non linear models
                non_linear_models_loaded = add_lags_to_dict(non_linear_models_loaded, config["shap"]["daily_base_non_linear"]["extracted_lags"])
               
                # We will only include linear_order_0 in this plot
                linear_order_0 = full_linear_models_loaded[f"{LINEAR_MODEL_PREFIX}0"]
                full_linear_models_loaded = {f"{LINEAR_MODEL_PREFIX}0": linear_order_0}

                # Steps for the forecast
                steps = daily_steps

                # Run forecasts
                run_forecasts_diff_lags(steps, full_linear_models_loaded, non_linear_models_loaded, True, "D", ts_daily_train, ts_daily_test)

            elif key == config["hyperparameter_tuning"]["daily_hybrid_key"]:
                # Load the previous non linear model
                hybrid_models_loaded, non_linear_models_loaded = load_models(config["model_sigs"]["daily_hybrid_reduced"])

                # Load the previous non linear design matrix
                hybrid_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["daily_hybrid_reduced"])

                # Define the model, **value2 will pass the dict of hyperparamters to the XGBRegressor 
                new_hybrid = xgb.XGBRegressor(
                    **value2
                )

                plotting_hybrid_models_loaded = {}

                # We will need to fit new_hybrid to the residuals of the model
                for i in [config["shap"]["daily_hybrid_order"]]: # We only fit to the order specified in the config file
                    # Unpack design, target and dp, we make deep copies to avoid any potential issues with references
                    X = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0])
                    y = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][1])
                    dp = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][2])

                    # Make a temporary copy of the hybrid model to avoid overwriting the original
                    tmp_hybrid = copy.deepcopy(new_hybrid)

                    # Convert to numpy arrays before prediction
                    X_pred = X.to_numpy(copy = True)

                    # Compute residuals
                    y_fit = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0].predict(X_pred)
                    y_resid = y - y_fit

                    # If using GPU convert to CuPy arrays and fit the model
                    if config["xgboost_setup"]["device"] == "cuda":
                        X_cp = cp.asarray(X)
                        y_resid_cp = cp.asarray(y_resid)
                        tmp_hybrid.fit(X_cp, y_resid_cp)
                    else:
                        # Fit hybrid to residuals
                        tmp_hybrid.fit(X, y_resid)
                    
                    # Add this hybrid model to the dict of plotting hybrid models along with the corresponding original model
                    plotting_hybrid_models_loaded[f"{key2}_{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"] = (hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0], dp, tmp_hybrid)
                    plotting_hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"] = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"]

                # Add lags to both dicts
                plotting_hybrid_models_loaded = add_lags_to_dict(plotting_hybrid_models_loaded, config["shap"]["daily_hybrid_order2"]["extracted_lags"])
                non_linear_models_loaded = add_lags_to_dict(non_linear_models_loaded, config["shap"]["daily_base_non_linear"]["extracted_lags"])

                # Steps for the forecast
                steps = daily_steps

                # Run forecasts
                run_forecasts_diff_lags(steps, plotting_hybrid_models_loaded, non_linear_models_loaded, True, "D", ts_daily_train, ts_daily_test)

            elif key == config["hyperparameter_tuning"]["hourly_linear_key"]:
                # Load the previous non linear model
                linear_models_loaded, non_linear_models_loaded = load_models(config["model_sigs"]["hourly_non_linear_reduced"])

                # Load the previous non linear design matrix
                linear_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["hourly_non_linear_reduced"])

                # Get design, target and dp
                X = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][0]
                y = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][1]
                dp = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][2]

                # Define the model, **value2 will pass the dict of hyperparamters to the XGBRegressor
                new_non_linear = xgb.XGBRegressor(
                    **value2
                )

                # Convert to CuPy arrays before fitting model
                X_cp = cp.asarray(X)
                y_cp = cp.asarray(y)

                # Fit the model
                new_non_linear.fit(X_cp,y_cp)

                # Add this non linear model to the dict of non linear models
                non_linear_models_loaded[key2] = (new_non_linear, dp, None)




                # Because we have two sets of lags we need to use run_forecasts_diff_lags, which means we have to add the lags to the loaded dicts

                # Now we want to load the full linear models (not the reduced ones)
                full_linear_models_loaded, full_non_linear_models_loaded = load_models(config["model_sigs"]["hourly_linear"])    

                # Add lags to full linear models 
                full_linear_models_loaded = add_lags_to_dict(full_linear_models_loaded, old_hourly_lags)

                # Add lags to reduced non linear models
                non_linear_models_loaded = add_lags_to_dict(non_linear_models_loaded, config["shap"]["hourly_base_non_linear"]["extracted_lags"])

                # We will only include linear_order_0 in this plot
                linear_order_0 = full_linear_models_loaded[f"{LINEAR_MODEL_PREFIX}0"]
                full_linear_models_loaded = {f"{LINEAR_MODEL_PREFIX}0": linear_order_0}

                # Steps for the forecast
                steps = hourly_steps

                # Run forecasts
                run_forecasts_diff_lags(steps, full_linear_models_loaded, non_linear_models_loaded, True, "h", ts_hourly_train, ts_hourly_test)

            elif key == config["hyperparameter_tuning"]["hourly_hybrid_key"]:
                # Load the previous non linear model
                hybrid_models_loaded, non_linear_models_loaded = load_models(config["model_sigs"]["hourly_hybrid_reduced"])

                # Load the previous non linear design matrix
                hybrid_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["hourly_hybrid_reduced"])

                # Define the model, **value2 will pass the dict of hyperparamters to the XGBRegressor 
                new_hybrid = xgb.XGBRegressor(
                    **value2
                )


                plotting_hybrid_models_loaded = {}

                # We will need to fit new_hybrid to the residuals of the model
                for i in [config["shap"]["hourly_hybrid_order"]]:
                    # Unpack design, target and dp, we make deep copies to avoid any potential issues with references
                    X = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][0])
                    y = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][1])
                    dp = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][2])

                    # Make a temporary copy of the hybrid model to avoid overwriting the original
                    tmp_hybrid = copy.deepcopy(new_hybrid)

                    # Convert to numpy arrays before prediction
                    X_pred = X.to_numpy(copy = True)

                    # Compute residuals
                    y_fit = hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][0].predict(X_pred)
                    y_resid = y - y_fit

                    # If using GPU convert to CuPy arrays and fit the model
                    if config["xgboost_setup"]["device"] == "cuda":
                        X_cp = cp.asarray(X)
                        y_resid_cp = cp.asarray(y_resid)
                        tmp_hybrid.fit(X_cp, y_resid_cp)
                    else:
                        # Fit hybrid to residuals
                        tmp_hybrid.fit(X, y_resid)
                    
                    # Add this hybrid model to the dict of plotting hybrid models along with the corresponding original model
                    plotting_hybrid_models_loaded[f"{key2}_{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"] = (hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][0], dp, tmp_hybrid)
                    plotting_hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"] = hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"]

                # Add lags to both dicts
                plotting_hybrid_models_loaded = add_lags_to_dict(plotting_hybrid_models_loaded, config["shap"]["hourly_hybrid_order2"]["extracted_lags"])
                non_linear_models_loaded = add_lags_to_dict(non_linear_models_loaded, config["shap"]["hourly_base_non_linear"]["extracted_lags"])



                # Steps for the forecast
                steps = hourly_steps

                # Run forecasts
                run_forecasts_diff_lags(steps, plotting_hybrid_models_loaded, non_linear_models_loaded, True, "h", ts_hourly_train, ts_hourly_test)

            else:
                raise ValueError("Key must be one of 'daily', 'daily_hybrid', 'hourly', 'hourly_hybrid'")

