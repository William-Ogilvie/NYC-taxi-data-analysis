"""
hyperparam_helpers
=================

This module contains helper functions for hyperparameter tuning using hyperopt. It includes both functions to make the validation folds and the objective function itself.
Along with a wrapper function to allow passing additional parameters to the objective function when using fmin from hyperopt.
"""

# --- Imports ---
import xgboost as xgb
from .forecast_helpers import forecast, run_forecasts, preprocess
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
import optuna

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
        # print(f"Fold {fold}")
        # print(train_index)
        # We need to preprocess the training portion of the fold
        ts_train = ts.iloc[train_index].copy()
        (X_train, y_train, dp, lags) = preprocess(lags, constant, order, fourier_features, time_step, ts_train)

        # We don't need to preprocess the test portion of the fold because we are going to pass the deterministic process and use dp.out_sample()
        # when forecasting as we are doing a multistep forecast and need to build lags as we go. 
        y_test = ts.iloc[test_index].copy()


        # To improve memory usage set to float32
        X_train = X_train.astype("float32")
        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")
        
        fold_dict[f"fold_{fold}"] = (X_train, y_train, dp, lags, y_test)

    return fold_dict

# # Hyperopt version
# def objective(space: dict, fold_dict: dict, lags: list[int], steps: int, hybrid: LinearRegression | None, offset_list: list[int]) -> dict:
#     """ Objective function for hyperparameter tuning using hyperopt

#     Args:
#         space (dict): hyperparameter space
#         fold_dict (dict): dict containing the folds with keys as fold_0, fold_1, ..., each value is a tuple (X_train, y_train, dp, y_test)
#         lags (list[int]): list of lags
#         steps (list[int]): number of steps to forecast
#         hybrid (LinearRegression | None): hybrid model to use, if None then no hybrid model is used
#         offset_list (list[int]): list of offsets to use when forecasting

#     Returns:
#         dict: dictionary containing the computed loss (mean MAE across folds) and the status 
#     """     

#     model = XGBRegressor(
#         n_estimators = space["n_estimators"],
        
#         learning_rate = space["learning_rate"],

#         max_depth = space["max_depth"],
#         min_child_weight = space["min_child_weight"],

#         subsample = space["subsample"],
#         colsample_bytree = space["colsample_bytree"],

#         reg_lambda = space["reg_lambda"],
#         reg_alpha = space["reg_alpha"],

#         gamma = space["gamma"],

#         random_state = space["random_state"],
#         #early_stopping_rounds = space["early_stopping_rounds"],
#         eval_metric = space["eval_metric"],
        

#         # Tree method hist will eseentially bin feature values into histograms and consider 
#         # and then only considers splits at bin boundaries. 
#         # If you have a gpu it is advised you use it particularly for training the hourly dataset, to do set device = "cuda" (you may need to install the gpu version of xgboost manually with conda install -c conda-forge py-xgboost=*=cuda*)
#         tree_method = space["tree_method"],
#         device = space["device"] # use gpu
#         )

#     # If we are in the hybrid case then the model above is actually the "hybrid" part used in forecasting, and we have passed the linear regression as the hybrid component of this function
#     # so we need to swap theses two
#     if hybrid is not None:
#         tmp = model
#         model = hybrid
#         hybrid = tmp

#     maes = []
#     for value in fold_dict.values():
#         X_train = value[0]
#         y_train = value[1]
#         dp = value[2]
#         lags = value[3]
#         y_test = value[4]
 

#         # We are going to use early stopping, now to avoid very long computations we will take a slight shorcut in that the validation set will have 
#         # the "real lags" rather than the lags computed during the forecast on the test set of the fold. 

#         # # Get the number of rows in X_train
#         # num_rows = X_train.shape[0]

#         # # We will validate on about 10% of the data
#         # split_row = int(num_rows * 0.9) # this tells us which row to split on
#         # X_val = X_train.iloc[split_row:]
#         # y_val = y_train.iloc[split_row:]
    

#         # X_train = X_train.iloc[:split_row]
#         # y_train = y_train.iloc[:split_row]

#         # Convert to numpy arrays before fitting model
#         X_train_np = X_train.to_numpy(copy = True)
#         y_train_np = y_train.to_numpy(copy = True)


#         # X_val_np = X_val.to_numpy(copy = True)
#         # y_val_np = y_val.to_numpy(copy = True)

#         # Time fitting the model
#         start_fit = time.time()

#         # If there is no hybrid model we can just fit the model to the data
#         if hybrid is None:
#             # Convert to CuPy arrays if using GPU
#             if config["xgboost_setup"]["device"] == "cuda":
#                 X_train_cp = cp.asarray(X_train_np)
#                 y_train_cp = cp.asarray(y_train_np)

#                 # Fit the model
#                 model.fit(X_train_cp, y_train_cp)
#             else:
#                 model.fit(X_train_np, y_train_np)

#         # If we have a hyrid model then this will need to be fit to the data first (as it is the linear part) and then we pass the XGBoost part to the forecast as hybrid
#         else: 
#             # Fit model to the data (we fit to the numpy arrays as model is a linear regression)
#             model.fit(X_train_np, y_train_np)

#             # We now need to compute the residuals and fit the hybrid model to these
#             y_fit = model.predict(X_train_np)
#             y_resid = y_train_np - y_fit

#             # Convert to CuPy arrays if using GPU and fit the model
#             if config["xgboost_setup"]["device"] == "cuda":
#                 X_train_cp = cp.asarray(X_train_np)
#                 y_resid_cp = cp.asarray(y_resid)
#                 hybrid.fit(X_train_cp, y_resid_cp)
#             else:
#                 hybrid.fit(X_train_np, y_resid)

#         end_fit = time.time()

#         # Time forecasting
#         start_fore = time.time()

#         # We want to use gpu for predictions if hybrid is None (so we are just using XGBoost) and the config file says to use cuda
#         if hybrid is None and config["xgboost_setup"]["device"] == "cuda":
#             gpu = True
#         else:
#             gpu = False 

#         # Run the forecast for the required steps on each of the offsets and take an average
#         mae_list = []
#         for offset in offset_list:
#             # Forecast on offset
#             y_preds = forecast(model, y_train, lags, steps, offset, dp, hybrid, gpu)

#             # Get the true values we are forecasting
#             y_test_offset = y_test.iloc[offset:offset+steps]

#             # Compute MAE for this offset 
#             mae = mean_absolute_error(y_preds, y_test_offset)

#             # Append to list
#             mae_list.append(mae)
        

#         end_fore = time.time()
    
#         # Report timings
#         print(f"Fit time: {end_fit - start_fit:.2f} seconds")
#         print(f"Predict time: {end_fore - start_fore:.4f} seconds")

#         # # See iterations stopped at
#         # print("Best iteration:", model.best_iteration)
#         # print("Best score:", model.best_score)

        
#         # Average the MAEs and add to list
#         print(mae_list)
#         mae = sum(mae_list) / len(mae_list) 
#         maes.append(mae) 

#     # Average MAE across folds (technically this is a double average as we have averaged across offsets too)
#     mean_mae = sum(maes) / len(maes)
#     print("MAEs:", maes)
#     print("Avg MAE:", mean_mae)
#     return {'loss': mean_mae, 'status': STATUS_OK}

def define_model(trial: optuna.trial.Trial) -> XGBRegressor:
    """ Defines the XGBoost model with hyperparameters from Optuna trial

    Args:
        trial (optuna.trial.Trial): Optuna trial object

    Returns:
        XGBRegressor: XGBoost model
    """    
    
    # Define search space
    space = {
    # Number of trees
    "n_estimators": config["xgboost_default"]["n_estimators"], # we use early stopping to select for n_estimators
    "early_stopping_rounds": config["xgboost_default"]["early_stopping_rounds"],

    # Learning rate
    # step size shrinkage, smaller = slower but more precise learning
    "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1, log = True),

    # Depth/complexity
    # Max depth of tree, larger more complex trees but can cause overfitting
    "max_depth": trial.suggest_int("max_depth", 3, 6, step = 1), # scope.int ensures we take ints only
    # minimum "weight" needed in child node. Higher values more conservative, fewer splits helps prevent overfitting
    "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0, log=True),

    # Randomisation/feature subsampling
    # fraction of rows used per tree, lower adds randomness reduces overfitting
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    # fraction of features used per tree
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

    # Regularisation
    # L2 penalty, good range is [0.1, 10] we use loguniform because this means that every order of magnitude has equal probability  
    "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True), # [0.01, 100]
    # L1 penalty
    "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 10.0, log=True), # [0.001, 10]

    # Split penalty (gamma) 
    # minimum loss reduction required to split a node, higher values = more conservative
    "gamma": trial.suggest_float("gamma", 0.0009, 10.0, log=True), # approx [0.0009, 10]

    "random_state": config["xgboost_setup"]["random_state"], 
    "eval_metric": config["xgboost_setup"]["eval_metric"], 
    "tree_method": config["xgboost_setup"]["tree_method"], 
    "device": config["xgboost_setup"]["device"] # Use GPU if available
    }

    model = XGBRegressor(
        n_estimators = space["n_estimators"],
        early_stopping_rounds = space["early_stopping_rounds"],
        
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
        device = space["device"] # use gpu or cpu
        )
    
    return model


def objective_optuna(trial: optuna.trial.Trial, fold_dict: dict, steps: int, hybrid: LinearRegression | None, offset_list: list[int]) -> float:
    """ Objective function for hyperparameter tuning using optuna

    Args:
        trial (optuna.trial.Trial): Optuna trial object
        fold_dict (dict): dict containing the folds with keys as fold_0, fold_1, ..., each value is a tuple (X_train, y_train, dp, y_test) 
        steps (list[int]): number of steps to forecast
        hybrid (LinearRegression | None): hybrid model to use, if None then no hybrid model is used
        offset_list (list[int]): list of offsets to use when forecasting

    Returns:
        float: mean maes across folds
    """     

    model = define_model(trial) 
  

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
        lags = value[3]
        y_test = value[4]

        # Create validation set for early stopping using 10% of training data
        num_rows = X_train.shape[0]
        split_row = int(num_rows * 0.9)

        X_val = X_train.iloc[split_row:]
        y_val = y_train.iloc[split_row:]
        X_train_fit = X_train.iloc[:split_row]
        y_train_fit = y_train.iloc[:split_row]

        # Convert to numpy arrays before fitting model
        X_train_np = X_train_fit.to_numpy(copy = True)
        y_train_np = y_train_fit.to_numpy(copy = True)
        X_val_np = X_val.to_numpy(copy=True)
        y_val_np = y_val.to_numpy(copy=True)

        # Time fitting the model
        start_fit = time.time()

        # If there is no hybrid model we can just fit the model to the data
        if hybrid is None:
            # Convert to CuPy arrays if using GPU
            if config["xgboost_setup"]["device"] == "cuda":
                X_train_cp = cp.asarray(X_train_np)
                y_train_cp = cp.asarray(y_train_np)
                X_val_cp = cp.asarray(X_val_np)
                y_val_cp = cp.asarray(y_val_np)

                # Fit the model
                model.fit(
                    X_train_cp, y_train_cp,
                    eval_set=[(X_val_cp, y_val_cp)],
                    verbose = False)
            else:
                model.fit(
                    X_train_np, y_train_np,
                    eval_set=[(X_val_np, y_val_np)],
                    verbose = False)

        # If we have a hyrid model then this will need to be fit to the data first (as it is the linear part) and then we pass the XGBoost part to the forecast as hybrid
        else: 
            # Fit model to the data (we fit to the numpy arrays as model is a linear regression)
            model.fit(X_train_np, y_train_np)

            # Compute residuals on training set 
            y_fit = model.predict(X_train_np)
            y_resid = y_train_np - y_fit

            # Compute residuals on validation set
            y_fit_val = model.predict(X_val_np)
            y_resid_val = y_val_np - y_fit_val

            # Convert to CuPy arrays if using GPU and fit the model
            if config["xgboost_setup"]["device"] == "cuda":
                X_train_cp = cp.asarray(X_train_np)
                y_resid_cp = cp.asarray(y_resid)
                X_val_cp = cp.asarray(X_val_np)
                y_resid_val_cp = cp.asarray(y_resid_val)
                hybrid.fit(
                    X_train_cp, y_resid_cp,
                    eval_set=[(X_val_cp, y_resid_val_cp)],
                    verbose = False)
            else:
                hybrid.fit(
                    X_train_np, y_resid,
                    eval_set=[(X_val_np, y_resid_val)],
                    verbose = False)

        end_fit = time.time()

        # Time forecasting
        start_fore = time.time()

        # We want to use gpu for predictions if hybrid is None (so we are just using XGBoost) and the config file says to use cuda
        if hybrid is None and config["xgboost_setup"]["device"] == "cuda":
            gpu = True
        else:
            gpu = False 

        # Run the forecast for the required steps on each of the offsets and take an average
        mae_list = []
        for offset in offset_list:
            # Forecast on offset
            y_preds = forecast(model, y_train, lags, steps, offset, dp, hybrid, gpu)

            # Get the true values we are forecasting
            y_test_offset = y_test.iloc[offset:offset+steps]

            # Compute MAE for this offset 
            mae = mean_absolute_error(y_preds, y_test_offset)

            # Append to list
            mae_list.append(mae)
        print(mae_list) 

        end_fore = time.time()
    
        # # Report timings
        # print(f"Fit time: {end_fit - start_fit:.2f} seconds")
        # print(f"Predict time: {end_fore - start_fore:.4f} seconds")

        # # See iterations stopped at
        # print("Best iteration:", model.best_iteration)
        # print("Best score:", model.best_score)

        
        # Average the MAEs and add to list
        # print(mae_list)
        mae = sum(mae_list) / len(mae_list) 
        maes.append(mae) 
    print(maes)

    # Average MAE across folds (technically this is a double average as we have averaged across offsets too)
    mean_mae = sum(maes) / len(maes)
    # print("MAEs:", maes)
    # print("Avg MAE:", mean_mae)
    return mean_mae

# def wrapped_objective_optuna(trial: optuna.trial.Trial) -> float:
#     """ Wrapper function for the objective to include additional parameters.

#     Args:
#         trial (optuna.trial.Trial): Optuna trial object
#     Returns:
#         float: same return as objective function, mean maes across folds
#     """
#     return objective_optuna(trial, wrapped_objective_optuna.fold_dict, wrapped_objective_optuna.steps, wrapped_objective_optuna.hybrid, wrapped_objective_optuna.offset_list)



# def wrapped_objective(space: dict) -> dict:
#     """ Wrapper function for the objective to include additional parameters.

#     Args:
#         space (dict): hyperparameter space

#     Returns:
#         dict: same return as objective function, dict of loss and status
#     """    
#     return objective(space, wrapped_objective.fold_dict, wrapped_objective.lags, wrapped_objective.steps, wrapped_objective.hybrid, wrapped_objective.offset_list)

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


def test_hyperparams(dict_full: dict, ts_daily_train: pd.Series, ts_daily_test: pd.Series, ts_hourly_train: pd.Series, ts_hourly_test: pd.Series, daily_steps: list[int], hourly_steps: list[int], daily_offsets: list[int], hourly_offsets: list[int], daily_offsets_to_show: list[int], hourly_offsets_to_show: list[int]) -> None:
    """ Tests the hyperparameters by loading the previous models and design matrices, defining a new model with the hyperparameters and fitting it to the training data.
        Outputs the same forecasts using run_forecasts as in the modelling notebook.

    Args:
        dict_full (dict): dict of dict of hyperparamters, split into four keys, "daily", "daily_hybrid", "hourly", "hourly_hybrid", then each dict has values of the hyperparameters key of the model signature 
        ts_daily_train (pd.Series): training data for the daily model
        ts_daily_test (pd.Series): test data for the daily model
        ts_hourly_train (pd.Series): training data for the hourly model
        ts_hourly_test (pd.Series): test data for the hourly model
        daily_steps (list[int]): list of steps to forecast for the daily model
        hourly_steps (list[int]): list of steps to forecast for the hourly model
    """    



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
                lags = non_linear_design_loaded[REDUCED_DAILY_DEFAULT_NON_LINEAR][3]

                # Define the model, **value2 will pass the dict of hyperparamters to the XGBRegressor 
                new_non_linear = xgb.XGBRegressor(
                    **value2 
                )

                # Create validation set for early stopping use 10% of training data
                num_rows = X.shape[0]
                split_row = int(num_rows * 0.9)

                X_train = X.iloc[:split_row]
                y_train = y.iloc[:split_row]
                X_val = X.iloc[split_row:]
                y_val = y.iloc[split_row:]   

                # Check if using gpu
                if config["xgboost_setup"]["device"] == "cuda":
                    # Convert to CuPy arrays before fitting model
                    X_train_cp = cp.asarray(X_train)
                    y_train_cp = cp.asarray(y_train)
                    X_val_cp = cp.asarray(X_val)
                    y_val_cp = cp.asarray(y_val)

                    # Fit the model
                    new_non_linear.fit(
                        X_train_cp, y_train_cp,
                        eval_set=[(X_val_cp, y_val_cp)],
                        verbose = False)
                else:
                    # Convert to Numpy arrays before fitting model
                    X_train_np = X_train.to_numpy(copy = True)
                    y_train_np = y_train.to_numpy(copy = True)
                    X_val_np = X_val.to_numpy(copy = True)
                    y_val_np = y_val.to_numpy(copy = True)

                    # Fit the model
                    new_non_linear.fit(
                        X_train_np, y_train_np,
                        eval_set=[(X_val_np, y_val_np)],
                        verbose = False)

                # Add this non linear model to the dict of non linear models
                non_linear_models_loaded[key2] = (new_non_linear, dp, None, lags)


                # Because we have two sets of lags we need to use run_forecasts_diff_lags, which means we have to add the lags to the loaded dicts

                # Now we want to load the full linear models (not the reduced ones)
                full_linear_models_loaded, full_non_linear_models_loaded = load_models(config["model_sigs"]["daily_linear"])    
 
                # We will only include linear_order_0 in this plot
                linear_order_0 = full_linear_models_loaded[f"{LINEAR_MODEL_PREFIX}0"]
                full_linear_models_loaded = {f"{LINEAR_MODEL_PREFIX}0": linear_order_0}

                # Steps for the forecast
                steps = daily_steps

                # Save file tag
                save_file_tag = config["saving"]["daily_non_linear_tuned"] + "_" + key2

                # Run forecasts
                run_forecasts(steps, daily_offsets, daily_offsets_to_show, full_linear_models_loaded, non_linear_models_loaded, True, "D", ts_daily_train, ts_daily_test, save_file_tag)

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
                    lags = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][3])


                    # Create validation set for early stopping use 10% of training data
                    num_rows = X.shape[0]
                    split_row = int(num_rows * 0.9)

                    X_train = X.iloc[:split_row]
                    y_train = y.iloc[:split_row]
                    X_val = X.iloc[split_row:]
                    y_val = y.iloc[split_row:]  

                    # Make a temporary copy of the hybrid model to avoid overwriting the original
                    tmp_hybrid = copy.deepcopy(new_hybrid)

                    # Convert to numpy arrays before prediction
                    X_train_np = X_train.to_numpy(copy = True)
                    X_val_np = X_val.to_numpy(copy = True)
                    y_train_np = y_train.to_numpy(copy = True)
                    y_val_np = y_val.to_numpy(copy = True)

                    # Compute residuals for both train and val sets
                    y_fit = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0].predict(X_train_np)
                    y_resid_np = y_train_np - y_fit
                    y_fit_val = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0].predict(X_val_np)
                    y_resid_val_np = y_val_np - y_fit_val
                     

                    # If using GPU convert to CuPy arrays and fit the model
                    if config["xgboost_setup"]["device"] == "cuda":
                        X_train_cp = cp.asarray(X_train_np)
                        y_resid_cp = cp.asarray(y_resid_np)
                        X_val_cp = cp.asarray(X_val_np)
                        y_resid_val_cp = cp.asarray(y_resid_val_np)

                        tmp_hybrid.fit(
                            X_train_cp, y_resid_cp,
                            eval_set=[(X_val_cp, y_resid_val_cp)],
                            verbose = False)
                    else:
                        
                        # Fit hybrid to residuals
                        tmp_hybrid.fit(
                            X_train_np, y_resid_np,
                            eval_set = [(X_val_np, y_resid_val_np)],
                            verbose = False)
                    
                    # Add this hybrid model to the dict of plotting hybrid models along with the corresponding original model
                    plotting_hybrid_models_loaded[f"{key2}_order{i}"] = (hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0], dp, tmp_hybrid, lags)
                    plotting_hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"] = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"]

                # Steps for the forecast
                steps = daily_steps

                # Save file tag
                save_file_tag = config["saving"]["daily_hybrid_tuned"] + "_" + key2

                # Run forecasts
                run_forecasts(steps, daily_offsets, daily_offsets_to_show, plotting_hybrid_models_loaded, non_linear_models_loaded, True, "D", ts_daily_train, ts_daily_test, save_file_tag)

            elif key == config["hyperparameter_tuning"]["hourly_linear_key"]:
                # Load the previous non linear model
                linear_models_loaded, non_linear_models_loaded = load_models(config["model_sigs"]["hourly_non_linear_reduced"])

                # Load the previous non linear design matrix
                linear_design_loaded, non_linear_design_loaded = load_design(config["model_sigs"]["hourly_non_linear_reduced"])

                # Get design, target and dp
                X = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][0]
                y = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][1]
                dp = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][2]
                lags = non_linear_design_loaded[REDUCED_HOURLY_DEFAULT_NON_LINEAR][3]

                # Define the model, **value2 will pass the dict of hyperparamters to the XGBRegressor
                new_non_linear = xgb.XGBRegressor(
                    **value2
                )

                # Create validation set for early stopping use 10% of training data
                num_rows = X.shape[0]
                split_row = int(num_rows * 0.9)

                X_train = X.iloc[:split_row]
                y_train = y.iloc[:split_row]
                X_val = X.iloc[split_row:]
                y_val = y.iloc[split_row:]   

                # Check if using gpu
                if config["xgboost_setup"]["device"] == "cuda":
                    # Convert to CuPy arrays before fitting model
                    X_train_cp = cp.asarray(X_train)
                    y_train_cp = cp.asarray(y_train)
                    X_val_cp = cp.asarray(X_val)
                    y_val_cp = cp.asarray(y_val)

                    # Fit the model
                    new_non_linear.fit(
                        X_train_cp, y_train_cp,
                        eval_set=[(X_val_cp, y_val_cp)],
                        verbose = False)

                else:
                    # Convert to Numpy arrays before fitting model
                    X_train_np = X_train.to_numpy(copy = True)
                    y_train_np = y_train.to_numpy(copy = True)
                    X_val_np = X_val.to_numpy(copy = True)
                    y_val_np = y_val.to_numpy(copy = True)

                    # Fit the model
                    new_non_linear.fit(
                        X_train_np, y_train_np,
                        eval_set=[(X_val_np, y_val_np)],
                        verbose = False)

                # Add this non linear model to the dict of non linear models
                non_linear_models_loaded[key2] = (new_non_linear, dp, None, lags)




                # Because we have two sets of lags we need to use run_forecasts_diff_lags, which means we have to add the lags to the loaded dicts

                # Now we want to load the full linear models (not the reduced ones)
                full_linear_models_loaded, full_non_linear_models_loaded = load_models(config["model_sigs"]["hourly_linear"])    

                # We will only include linear_order_0 in this plot
                linear_order_0 = full_linear_models_loaded[f"{LINEAR_MODEL_PREFIX}0"]
                full_linear_models_loaded = {f"{LINEAR_MODEL_PREFIX}0": linear_order_0}

                # Steps for the forecast
                steps = hourly_steps

                # Save file tag
                save_file_tag = config["saving"]["hourly_non_linear_tuned"] + "_" + key2

                # Run forecasts
                run_forecasts(steps, hourly_offsets, hourly_offsets_to_show, full_linear_models_loaded, non_linear_models_loaded, True, "h", ts_hourly_train, ts_hourly_test, save_file_tag)

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
                    lags = copy.deepcopy(hybrid_design_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][3])

                    # Create validation set for early stopping use 10% of training data
                    num_rows = X.shape[0]
                    split_row = int(num_rows * 0.9)

                    X_train = X.iloc[:split_row]
                    y_train = y.iloc[:split_row]
                    X_val = X.iloc[split_row:]
                    y_val = y.iloc[split_row:]  

                    # Make a temporary copy of the hybrid model to avoid overwriting the original
                    tmp_hybrid = copy.deepcopy(new_hybrid)

                    # Convert to numpy arrays before prediction
                    X_train_np = X_train.to_numpy(copy = True)
                    X_val_np = X_val.to_numpy(copy = True)
                    y_train_np = y_train.to_numpy(copy = True)
                    y_val_np = y_val.to_numpy(copy = True)

                    # Compute residuals for both train and val sets
                    y_fit = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0].predict(X_train_np)
                    y_resid_np = y_train_np - y_fit
                    y_fit_val = hybrid_models_loaded[f"{REDUCED_HYBRID_DAILY_MODEL_PREFIX}{i}"][0].predict(X_val_np)
                    y_resid_val_np = y_val_np - y_fit_val

                    # If using GPU convert to CuPy arrays and fit the model
                    if config["xgboost_setup"]["device"] == "cuda":
                        X_train_cp = cp.asarray(X_train_np)
                        y_resid_cp = cp.asarray(y_resid_np)
                        X_val_cp = cp.asarray(X_val_np)
                        y_resid_val_cp = cp.asarray(y_resid_val_np)

                        tmp_hybrid.fit(
                            X_train_cp, y_resid_cp,
                            eval_set=[(X_val_cp, y_resid_val_cp)],
                            verbose = False)
                    else:
                        # Fit hybrid to residuals
                        tmp_hybrid.fit(
                            X_train_np, y_resid_np,
                            eval_set = [(X_val_np, y_resid_val_np)],
                            verbose = False)
                   
                    # Add this hybrid model to the dict of plotting hybrid models along with the corresponding original model
                    plotting_hybrid_models_loaded[f"{key2}_order{i}"] = (hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"][0], dp, tmp_hybrid, lags)
                    plotting_hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"] = hybrid_models_loaded[f"{REDUCED_HYBRID_HOURLY_MODEL_PREFIX}{i}"]



                # Steps for the forecast
                steps = hourly_steps

                # Save file tag
                save_file_tag = config["saving"]["hourly_hybrid_tuned"] + "_" + key2

                # Run forecasts
                run_forecasts(steps, hourly_offsets, hourly_offsets_to_show, plotting_hybrid_models_loaded, non_linear_models_loaded, True, "h", ts_hourly_train, ts_hourly_test, save_file_tag)

            else:
                raise ValueError("Key must be one of 'daily', 'daily_hybrid', 'hourly', 'hourly_hybrid'")

