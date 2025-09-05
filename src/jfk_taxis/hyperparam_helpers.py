from hyperopt import STATUS_OK
import xgboost as xgb
from jfk_taxis import preprocess, forecast
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import time
import cupy as cp
import pandas as pd
from xgboost import XGBRegressor

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


def objective(space: dict, fold_dict: dict, lags: list[int], steps: int, hybrid: XGBRegressor | None) -> dict:
    """ Objective function for hyperparameter tuning using hyperopt

    Args:
        space (dict): hyperparameter space
        fold_dict (dict): dict containing the folds with keys as fold_0, fold_1, ..., each value is a tuple (X_train, y_train, dp, y_test)
        lags (list[int]): list of lags
        steps (list[int]): number of steps to forecast 
        hybrid (XGBRegressor | None): hybrid model to use, if None then no hybrid model is used

    Returns:
        dict: dictionary containing the computed loss (mean MAE across folds) and the status 
    """     

    model = xgb.XGBRegressor(
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


    maes = []
    for value in fold_dict.values():
        X_train = value[0]
        y_train = value[1]
        dp = value[2]
        y_test = value[3]

        print(X_train.shape)

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
            X_train_cp = cp.asarray(X_train_np)
            y_train_cp = cp.asarray(y_train_np)

            # Fit the model
            model.fit(X_train_cp, y_train_cp)
        # If we have a hyrid model then this will need to be fit to the data first (as it is the linear part) and then we pass the XGBoost part to the forecast as hybrid
        else:
            # Swap model and hybrid
            tmp = model
            model = hybrid
            hybrid = tmp

            # Fit model to the data (we fit to the numpy arrays as model is a linear regression)
            model.fit(X_train_np, y_train_np)

        end_fit = time.time()

        # Time forecasting
        start_fore = time.time()
        
        # If there is no hybrid model then we want to use the gpu when forecasting otherwise we don't want to use the gpu (because we will get an error when linear regression tries to predict with a CuPy array)
        if hybrid is not None:
            gpu = True
        else:
            gpu = False

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
