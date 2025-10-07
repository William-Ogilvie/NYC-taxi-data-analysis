"""
forecast_helpers
=================

This module contains helper functions for forecasting the time series. The key ones are preprocess which create the design matrix, target vector and deterministic process.
Forecast which uses a trained model to forecast future values given a historical time series and a deterministic process.
Finally run_forecasts which will run forecasts for both linear (or potentially hybrid, meaning residual boosted) and non-linear models and create plots of both the forecasts
and the MAE scores for each model.
"""

# --- Imports ---
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import seaborn as sns
from statsmodels.tsa.deterministic import CalendarFourier, CalendarSeasonality
import numpy as np
import cupy as cp
import yaml
import copy
from .loading_helpers import load_config
from IPython.display import display

# --- Load config ---
config, PROJECT_ROOT = load_config()

# --- Functions ---
def drop_time_zone(s: pd.Series) -> pd.Series:
    """ Drop the timezone from a pandas Series index. Note this only works if the timezone is UTC (which it will be as we do all computations in UTC).

    Args:
        s (pd.Series): the input time series.
    Returns:
        pd.Series: the time series with timezone dropped.
    """
    s.index = s.index.tz_localize(None)
    return s

def preprocess(lags: list[int], constant: bool, order: int, fourier_features: list[str], time_step: str, ts: pd.Series) -> tuple[pd.DataFrame, pd.Series, DeterministicProcess, list[int]]:
    """ Preprocess the time series data for modeling.

    Args:
        lags (list[int]): list of lags to use
        constant (bool): bool for whether the deterministic process should have a constant
        order (int): order of the trend in the deterministic process
        fourier_features (list[str]): list of fourier features to use
        time_step (str): time step of the time series (e.g. "h", "D")
        ts (pd.Series): time series data

    Returns:
        tuple[pd.DataFrame, pd.Series, DeterministicProcess, list[int]]: design matrix, target series, the deterministic process fitted on the data and a list of the lags
    """    

    y = copy.deepcopy(ts) # Create a separate copy of the time series to avoid any changes to the original

    # Deterministic processes need to be passed a time zone naive index
    y = drop_time_zone(y)

    # When forecasting we need the index to have a frequency 
    y = y.asfreq(time_step)

    # # This may create some NaNs so we fill them with 0
    y = y.fillna(0)

    fourier_list = []
    # Fourier features for seasonality
    for feature in fourier_features:
        if feature == "YE":
            fourier_list.append(CalendarFourier(freq = "YE", order = 10)) # Annual seasonality (10 harmonics)
        elif feature == "W":
            fourier_list.append(CalendarFourier(freq = "W", order = 5)) # Weekly seasonality (5 harmonics)
        elif feature == "D":
            fourier_list.append(CalendarFourier(freq = "D", order = 5)) # Daily seasonality (5 harmonics)
   
   
    dp = DeterministicProcess(
        index = y.index,
        constant = constant,   # Dummy feature for bias (y-intercept)
        order = order,         # Polynomial trend (degree 1 = linear)
        seasonal = False,    # Don't use seasonal dummies
        additional_terms = fourier_list, # Add in the Fourier terms and any other extra features
        drop = False,       
    )

    X = dp.in_sample()

    # We now add in the lag features.
    # The reason we haven't used all the significant lags is we will need to drop the rows that
    # contain null values and if we use lag say 49 we will be dropping about 15% of our data

    # For performance reasons its better to make all lags at once and then concatante
    lag_cols = [y.shift(i).rename(f"y_lag_{i}") for i in lags] 
    X = pd.concat([X] + lag_cols, axis = 1)

    # Drop all na rows
    mask = X.notna().all(axis=1) # keep only rows with no NaNs
    X = X.loc[mask]
    y = y.loc[mask]

    # df = pd.concat([X, y], axis = 1)    
    # df = df.dropna()
    # y = df.iloc[:, -1] # target last col
    # X = df.iloc[:, :-1] # features all but last col


    return (X, y, dp, lags)
        

# Fit models


def to_numpy(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """ Helper to convert design and target to numpy arrays for model fitting

    Args:
        X (pd.DataFrame): design matrix
        y (pd.Series): target vector

    Returns:
        tuple[np.ndarray, np.ndarray]: design matrix and target vector as numpy arrays
    """

    X_np = X.to_numpy(copy = False)
    y_np = y.to_numpy(copy = False)
    return (X_np, y_np)

def fit_linear(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """ Fits linear regression to the design matrix and time series

    Args:
        X (pd.DataFrame): design matrix
        y (pd.Series): target vector

    Returns:
        LinearRegression: fitted linear regression model
    """

    (X, y) = to_numpy(X, y)
    model = LinearRegression(fit_intercept = False)
    model.fit(X,y)

    return model

def fit_non_linear(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """ Fits XGBoost to the design matrix and target vector

    Args:
        X (pd.DataFrame): design matrix
        y (pd.Series): target vector

    Returns:
        XGBRegressor: fitted XGBoost model
    """

    # Convert to numpy arrays
    (X, y) = to_numpy(X, y)
    
    # XGBoost:
    model_xgb = XGBRegressor(
        n_estimators=config["xgboost_default"]["n_estimators"],
        learning_rate=config["xgboost_default"]["learning_rate"],
        max_depth=config["xgboost_default"]["max_depth"],
        subsample=config["xgboost_default"]["subsample"],
        colsample_bytree=config["xgboost_default"]["colsample_bytree"],
        random_state=config["xgboost_setup"]["random_state"],
        eval_metric=config["xgboost_setup"]["eval_metric"],
        tree_method=config["xgboost_setup"]["tree_method"],
        device=config["xgboost_setup"]["device"]
    )
    
    # If using GPU convert to CUPy arrays 
    if config["xgboost_setup"]["device"] == "cuda":
        X = cp.asarray(X)
        y = cp.asarray(y)

    # Fit model
    model_xgb.fit(X, y)

    return model_xgb

def truncate_lags(lags: list[int], truncate_to: int) -> list[int]:
    """ Truncates the lags to the length of the historical data if the largest lag is greater than the length of the historical data

    Args:
        lags (list[int]): list of lags
        truncate_to (int): length of the historical data

    Returns:
        list[int]: truncated list of lags
    """

    lags = [lag for lag in lags if lag <= truncate_to]
   
    return lags

def to_NYC(s: pd.Series, freq: str) -> pd.Series:
    """Convert a pandas Series to NYC hourly, so we can convert back to local time for display.

    Args:
        s (pd.Series): the input time series.
        freq (str): frequency of the time series (e.g. "h", "D")

    Returns:
        pd.Series: the converted time series.
    """    

    # We likely have to localise to UTC first as loading from the csv will have removed the timezone info (although it still has the + 00:00 stamp)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")

    return s.tz_convert("America/New_York").asfreq(freq)

def forecast(model: LinearRegression | XGBRegressor, y: pd.Series, lags: list[int], steps: int, offset: int, dp: DeterministicProcess, hybrid: XGBRegressor | None, gpu: bool):
    """ Forecast future values using the trained model.

    Args:
        model (LinearRegression | XGBRegressor): trained model for forecasting.
        y (pd.Series): historical target values.
        lags (list[int]): list of lagged features to use.
        steps (int): number of steps to forecast.
        offset (int): offset to start the forecast from in the test values.
        dp (DeterministicProcess): deterministic process for generating future features.
        hybrid (XGBRegressor | None): optional hybrid model for boosting predictions.
        gpu (bool): whether to use GPU for predictions.

    Raises:
        IndexError: If lag buffer is too short for requested lag.

    Returns:
        pd.Series: Forecasted values.
    """
 
   
    preds = []
    y_hist = y.copy()

    # Create the deterministic features for the forecast
    X_future_det = dp.out_of_sample(steps = steps + offset)

    # Apply offset to X_future_det
    X_future_det = X_future_det.iloc[offset:]

    # display(X_future_det.head())
    # display(y_hist.tail())

    # Deteriministic features
    det_cols = X_future_det.columns.tolist()
    det_vals = X_future_det.to_numpy(copy = False) # shape : (steps, n_det)

    # Create lag column names
    lag_cols = [f'y_lag_{j}' for j in lags]
    feature_cols = det_cols + lag_cols

    # There is a potential issue if the historical data has less points than the largest lag, the solution will have to be to truncate
    # the lags to the length of the histrical data 
    if lags[-1] > len(y_hist):
        lags = truncate_lags(lags, len(y_hist))
        

    # Build lag buffer from last lags[-1] points
    last_lag = lags[-1]
    lag_buf = y_hist.iloc[-last_lag:].tolist()



    # Output array
    preds = np.empty(steps, dtype = np.float64)

    # Number of deterministic features
    n_det = det_vals.shape[1]

    # Create array to hold one row of features
    xrow = np.empty(n_det + len(lags), dtype = np.float64)

    for i in range(steps):
        # Deterministic part
        xrow[:n_det] = det_vals[i, :]

        # Lag part
        for j, lag in enumerate(lags):
            if len(lag_buf) < lag:
                print(f"lag_buf length: {len(lag_buf)}, lag: {lag}, lags: {lags}")
                raise IndexError("lag_buf too short for requested lag")
            xrow[n_det + j] = lag_buf[-lag]

        # Predict
        if gpu:
            xrow = cp.asarray(xrow)
            y_pred = model.predict(xrow.reshape(1, -1))[0]
            xrow = cp.asnumpy(xrow) # move back to numpy
        else:
            y_pred = model.predict(xrow.reshape(1, -1))[0]
             
        # Add hybrid prediction if applicable
        if hybrid is not None:

            # Check whether we are using GPU for hybrid model
            if config["xgboost_setup"]["device"] == "cuda":
                xrow = cp.asarray(xrow) 
                y_pred += hybrid.predict(xrow.reshape(1, -1))[0]
                y_pred = cp.asnumpy(y_pred) # move back to numpy
                xrow = cp.asnumpy(xrow) # move back to numpy
            else:
                y_pred += hybrid.predict(xrow.reshape(1, -1))[0]


        # Store prediction
        preds[i] = y_pred
        
        # Advance lag buffer
        lag_buf.append(y_pred)
        lag_buf.pop(0)

    return pd.Series(preds, index = X_future_det.index, name = getattr(y, "name"))


""" Old method using pandas only removed as inefficient
    # We first store the last lags[-1] of y_hist to use for lags
    lag_vals = list(y_hist.iloc[-lags[-1]:].copy())


    for i in range(steps): 

        # Get the deterministic row
        x_next = X_future_det.iloc[i].copy()
        
        # Create the lags using historical data
        # for j in lags:
        #     x_next[f'y_lag_{j}'] = y_hist.iloc[-j]
        #lag_dict = {f'y_lag_{j}': y_hist.iloc[-j] for j in lags}

        for j in lags:
            x_next[f'y_lag_{j}'] = lag_vals[-j]
        #x_next = pd.concat([x_next, pd.Series(lag_dict)], axis = 0)

        # lag_cols = [y_hist.shift(i).rename(f"y_lag_{i}") for i in lags] 
        # x_next = pd.concat([x_next] + lag_cols, axis = 1)

        # display(x_next.head())

        
        # Predict - x_next is a pandas series and needs to be converted to a dataframe for predictions
        y_pred = model.predict(pd.DataFrame([x_next], columns = x_next.index))[0]

        # If hybrid model add the hybrid models prediction to the linear prediction
        if hybrid is not None:
            y_pred += hybrid.predict(pd.DataFrame([x_next], columns = x_next.index))[0]

        
        # Append prediction to history so it can be used for future lags
        new_point = pd.Series(y_pred, index=[X_future_det.index[i]])
        new_point.index = pd.to_datetime(new_point.index) # ensure datetime index

        # Add this new prediction to the end of the lag_vals array
        lag_vals.append(int(new_point.iloc[0]))
         
        # old method concatenating to y_hist 
        #y_hist = pd.concat([y_hist, new_point])

        # Add prediction to preds series
        preds.append(new_point)

    # Turn preds into a pandas series
    preds = pd.concat(preds)
    return preds
"""

class MAEScore:
    """Container for MAE score of a specific forecast.

    This class stores a MAE score along with the model name, forecast step and the offset.

    Attributes:
        name (str): name of the model.
        mae (float): mean Absolute Error score.
        step (int): forecast step.
        offset (int): offset used in the forecast.
    """    
    def __init__(self, name: str, mae: float, step: int, offset: int) -> None:
        """Initialize the MAE score container.

        Args:
            name (str): name of the model.
            mae (float): mean Absolute Error score.
            step (int): forecast step.
            offset (int): offset used in the forecast.
        """        
        self.name = name
        self.mae = mae
        self.step = step
        self.offset = offset

class ModelMAEScores:
    """Container for MAE scores of a specific model.

    This class stores the model name and a list of mae_score objects for that model.

    Attributes:
        name (str): name of the model.
        scores (list[mae_score]): list of mae_score objects. 
    """    

    def __init__(self, name: str) -> None:
        """Initialize the model MAE scores container.

        Args:
            name (str): name of the model.
        """        
        self.name = name
        self.scores = []

    def append_score(self, score: MAEScore) -> None:
        """Append a MAE score to the model's score list.

        Args:
            score (mae_score): MAE score to append.
        """        
        self.scores.append(score)

    def average_mae_by_step(self) -> pd.Series:
        """Compute the average MAE by forecast step.

        Returns:
            pd.Series: series containing the average MAE for each forecast step.
        """        

        # First get all the unique steps
        steps = [score.step for score in self.scores]
        unique_steps = list(set(steps))
 
        mae_dict = {}

        for step in unique_steps:
            # Get all the MAE scores for this model and step
            step_scores = [score.mae for score in self.scores if score.step == step]

            # Compute the average MAE for this model and step
            avg_mae = sum(step_scores) / len(step_scores)

            # Store the average MAE in the dict
            mae_dict[step] = avg_mae

        # Convert the dict to a pandas series
        mae_series = pd.Series(mae_dict, name = self.name)

        
        return mae_series

def save_mae_scores(model_mae_list: dict, mae_scores: dict, step: int, offset: int) -> dict:
    """Save MAE scores to the model_mae_list.

    Args:
        model_mae_list (dict): list of model_mae_scores objects.
        mae_scores (dict): dictionary of MAE scores.
        step (int): forecast step.
        offset (int): offset used in the forecast.

    Returns:
        dict: updated model_mae_list.
    """    
    for name, mae_score in mae_scores.items():
        score = MAEScore(name, mae_score, step, offset) 
        model_mae_list[name].append_score(score)

    return model_mae_list

def create_avg_mae_df(model_mae_list: dict, linear_models: dict, non_linear_models: dict, naive: bool) -> pd.DataFrame:
    """ Create a dataframe of average MAE scores by forecast step for all models.

    Args:
        model_mae_list (dict): list of model_mae_scores objects.
        linear_models (dict): linear models dict
        non_linear_models (dict): non linear models dict
        naive (bool): whether the naive model is included

    Raises:
        ValueError: erorr if we can't find the model name in the model_mae_list

    Returns:
        pd.DataFrame: dataframe where the indicies are the steps, columns the model names and values the average MAE scores.
    """    

    # Find all unique model names
    model_names = list(set(linear_models.keys()).union(set(non_linear_models.keys())))

    # Add naive if used
    if naive:
        model_names.append("Naive")

    # Dataframe to hold the average MAE scores
    df_mae = pd.DataFrame()

    # Loop over all model names and get the average MAE scores by step
    for name in model_names:
        if name not in model_mae_list:
            raise ValueError(f"Model name {name} not found in model_mae_list")

        mae_series = model_mae_list[name].average_mae_by_step()

        df_mae[name] = mae_series
    
    return df_mae

def create_avg_mae_barplot(df_avg_mae: pd.DataFrame) -> plt.Figure:
    """ Create a bar plot of average MAE scores by forecast step for all models.

    Args:
        df_avg_mae (pd.DataFrame): dataFrame containing average MAE scores.

    Returns:
        plt.Figure: bar plot figure.
    """    
    # Make the data frame long format for a barplot
    df = df_avg_mae.reset_index()

    # make subplots with one column and len(df) rows
    fig, axes = plt.subplots(nrows = len(df), ncols = 1, figsize = (10,  6 * len(df)))

    if len(df) == 1:
        axes = [axes]
    
    for (i, row), ax in zip(df.iterrows(), axes):

        steps = int(row["index"])
        df_long = row.drop("index").reset_index()
        df_long.columns = ["model", "mae"]

        sns.barplot(
            data = df_long,
            x = "model",
            y = "mae",
            hue = "model",
            dodge = False,
            legend = False,
            ax = ax
        )
        
        ax.set_title(f"Average MAE for step {steps}")
        ax.tick_params(axis = "x", rotation = 90)

        # Print the average MAE scores for this step
        print(f"\nAverage MAE scores for step {steps}:")
        for row in df_long.itertuples():
            print(f"{row.model}, average MAE: {row.mae:.2f}")
    
    plt.tight_layout() 

    avg_bar_plot_fig = plt.gcf()
    plt.close()
    return avg_bar_plot_fig
    


    

def forecast_dicts(steps: list[int], y_test: pd.Series, y_hist: pd.Series, offset_list: list[int], offsets_to_show: list[int], linear_models: dict, non_linear_models: dict, naive: bool, time_step: str) -> None:
    """ Forecasting function that handles both linear and non-linear models, forecasts for each value in steps, computes the MAE and 
    creates both a plot of the forecast and a bar plot of the MAEs (MAEs are also printed as well).

    Args:
        steps (list[int]): list of steps to forecast.
        y_test (pd.Series): series of true future values.
        y_hist (pd.Series): series of historical values.
        offset_list (list[int]): offsets to start the forecast from in the test values.
        offset_to_show (list[int]): offsets to display the forecasts for.
        linear_models (dict): dictionary of linear models.
        non_linear_models (dict): dictionary of non-linear models.
        naive (bool): whether to include naive forecast.
        time_step (str): time step of the time series (e.g. "h", "D")
    """

    # Initalise instances of the model_mae_scores class for each model
    # List of model_mae_scores instances
    model_mae_list = {}
    for name, value in linear_models.items():
        model_mae = ModelMAEScores(name)
        model_mae_list[name] = model_mae

    for name, value in non_linear_models.items():
        model_mae = ModelMAEScores(name)
        model_mae_list[name] = model_mae

    # If using the naive model add it to the model_mae_list
    if naive:
        model_mae = ModelMAEScores("Naive")
        model_mae_list["Naive"] = model_mae

    # Loop over offsets
    for offset in offset_list:
        # Create a copy of y_hist and y_test to avoid any changes to the original
        y_test_copy = copy.deepcopy(y_test)
        y_hist_copy = copy.deepcopy(y_hist)

        # Apply offset to y_hist_copy and y_test_copy 
        y_hist_copy = pd.concat([y_hist_copy, y_test_copy.iloc[:offset]])
        y_test_copy = y_test_copy.iloc[offset:]



        # Compute naive predictions
        # Today = yesterday
        y_pred_naive = y_test_copy.shift(1)
        y_pred_naive.iloc[0] = y_hist_copy.iloc[-1]


        # Loop over steps
        for step in steps:

            # Store MAE scores for barplot and for the model_mae_list
            mae_scores = {}
            
            # Get real values
            y_real = y_test_copy.iloc[0:step]
            
            # Plot 
            y_real_plot = to_NYC(y_real, time_step)
            if time_step == "h":
                ax = y_real_plot.plot(color='0.25', style='.', title=f"Forecast steps: {step}, start date {y_real_plot.index[0]}, offset: {offset}")
            else: 
                ax = y_real_plot.plot(color='0.25', style='.', title=f"Forecast steps: {step}, start date {str(y_real_plot.index[0].date())}, offset: {offset}")
            # Check if there are any linear models to forecast
            if len(linear_models) != 0:
                # Forecast the linear models:
                for name, value in linear_models.items():
                    model = value[0]
                    dp = value[1]
                    hybrid = value[2]
                    lags = value[3]
                    
                    # Get forecast (we use cpu for linear forecasts, hence set gpu = False)
                    y_fore_linear = forecast(model, y_hist_copy, lags, step, offset, dp, hybrid, False)

                    
                    # Compute MAE linear
                    mae_linear = mean_absolute_error(y_fore_linear, y_real)
                    mae_scores[name] = mae_linear

                    # Only display for offsets in offsets_to_show
                    if offset in offsets_to_show: 
                        print(f"MAE: {mae_linear:.2f} for step = {step}, model = {name}")

                    # Add to plot
                    # Convert to NYC time for plotting
                    y_fore_linear = to_NYC(y_fore_linear, time_step)

                    ax = y_fore_linear.plot(ax = ax, label = name)
            

            # check if there are any non linear models to forecast
            if len(non_linear_models) != 0:
                # Forecast the non linear models::
                for name, value in non_linear_models.items():
                    model = value[0]
                    dp = value[1]
                    hybrid = value[2]
                    lags = value[3]

                    # Check whether using gpu
                    if config["xgboost_setup"]["device"] == "cuda":
                        gpu = True
                    else:
                        gpu = False

                    # Get forecast
                    y_fore_non_linear = forecast(model, y_hist_copy, lags, step, offset, dp, hybrid, gpu)

                    # Compute MAE non linear
                    mae_non_linear = mean_absolute_error(y_fore_non_linear, y_real)
                    mae_scores[name] = mae_non_linear

                    if offset in offsets_to_show:   
                        print(f"MAE: {mae_non_linear:.2f} for step = {step}, model = {name}")

                    # Add to plot
                    # Convert to NYC time for plotting
                    y_fore_non_linear = to_NYC(y_fore_non_linear, time_step)
                       
                    ax = y_fore_non_linear.plot(ax = ax, label = name)
        

            
            if naive == True:
                # Compute naive MAE
                y_step_pred_naive = y_pred_naive.loc[y_real.index]
            
                mae_naive = mean_absolute_error(y_real, y_step_pred_naive)
                mae_scores["Naive"] = mae_naive

                if offset in offsets_to_show:
                    print(f"MAE: {mae_naive:.2f} for step =  30, model = Naive\n")

                # Plot forecasts
                # Convert to NYC time for plotting
                y_step_pred_naive = to_NYC(y_step_pred_naive, time_step)

                ax = y_step_pred_naive.plot(ax = ax, label = "Naive")
                
            
                
            # Save the mae scores to the model_mae_list
            model_mae_list = save_mae_scores(model_mae_list, mae_scores, step, offset)
            
            # Add legend
            ax.legend()
            ax.set_ylabel("Trip counts")
            ax.set_xlabel("Pickup datetime")
            plt.xticks(rotation = 90, ha = "right")

            # This is for readability not necessarily efficiency
            if offset in offsets_to_show:
                plt.show()
            else:
                plt.close()
            # Plot MAE bar plots:
            df_mae = pd.DataFrame(list(mae_scores.items()), columns=["Model", "MAE"]) 

            plt.figure(figsize=(8,5))
            sns.barplot(data=df_mae, x="Model", y="MAE", hue = "Model")

            if time_step == "h":
                # Use the NYC time for title 
                plt.title(f"Model Comparison by MAE, steps = {step}, start date = {y_real_plot.index[0]}, offset = {offset}")
            else:
                plt.title(f"Model Comparison by MAE, steps = {step}, start date = {str(y_real_plot.index[0].date())}, offset = {offset}")
            plt.xticks(rotation=90, ha="right")


            if offset in offsets_to_show:
                plt.show()
            else:
                plt.close()

    # Now calculate the average MAE by step for each model and put into a dataframe
    df_avg_mae = create_avg_mae_df(model_mae_list, linear_models, non_linear_models, naive)  

    # Create bar plot of average MAE by step for each model for all steps
    bar_plot_fig = create_avg_mae_barplot(df_avg_mae)
    display(bar_plot_fig)
    
def run_forecasts(steps: list[int], offset_list: list[int], offsets_to_show: list[int], linear_models: dict, non_linear_models: dict, naive: bool, time_step: str, old_ts: pd.Series, new_ts: pd.Series) -> None:
    """ Run forecasts for both linear and non-linear models with the option of a naive baseline.

    Args:
        steps (list[int]): list of steps to forecast
        offset_list (list[int]): offsets to start the forecast from in the test values.
        offsets_to_show (list[int]): offsets to display the forecasts for
        linear_models (dict): dict of linear models
        non_linear_models (dict): dict of non linear models
        naive (bool): bool of whether to include naive baseline
        time_step (str): time step for the time series (e.g. "h", "D")
        old_ts (pd.Series): historical time series
        new_ts (pd.Series): future time series to compare forecasts against
    """   

    y_test = copy.deepcopy(new_ts) # Create deepcopys to avoid any changes to the original
    y_hist = copy.deepcopy(old_ts) 

    forecast_dicts(steps, y_test, y_hist, offset_list, offsets_to_show, linear_models, non_linear_models, naive, time_step)

def run_forecasts_app(steps: int, offset_list: list[int], linear_models: dict, non_linear_models: dict, naive: bool, time_step: str, old_ts: pd.Series, new_ts: pd.Series) -> tuple[dict, dict, plt.Figure]:
    """ Run forecasts for both linear and non-linear models with the option of a naive baseline, this is the app version the only differnece is we return the figure rather than showing it.

    Args:
        steps (int): steps to forecast
        offset_list (list[int]): offsets to start the forecast from in the test values.
        linear_models (dict): dict of linear models
        non_linear_models (dict): dict of non linear models
        naive (bool): bool of whether to include naive baseline
        time_step (str): time step for the time series (e.g. "h", "D")
        old_ts (pd.Series): historical time series
        new_ts (pd.Series): future time series to compare forecasts against

    Returns:
        tuple[dict, dict, plt.Figure]: dictionary of forecast figures, dictionary of bar plot figures, average bar plot figure
    """    


    y_test = copy.deepcopy(new_ts) # Create deepcopys to avoid any changes to the original
    y_hist = copy.deepcopy(old_ts)
    y_hist.index = pd.to_datetime(y_hist.index) 
   # y_hist.index = pd.date_range(start=y_hist.index[0], periods=len(y_hist), freq=time_step)
    
    forecast_figs, bar_plot_figs, avg_bar_plot_fig = forecast_dicts_app(steps, y_test, y_hist, offset_list, linear_models, non_linear_models, naive, time_step)

    return forecast_figs, bar_plot_figs, avg_bar_plot_fig


def forecast_dicts_app(steps: int, y_test: pd.Series, y_hist: pd.Series, offset_list: list[int], linear_models: dict, non_linear_models: dict, naive: bool, time_step: str) -> tuple[dict, dict, plt.figure]:
    """ Forecasting function that handles both linear and non-linear models, forecasts for each value in steps, computes the MAE and 
    creates both a plot of the forecast and a bar plot of the MAEs (MAEs are also printed as well). This is a modified version for the app.
    Only does one step at a time and returns the figures rather than showing them.

    Args:
        steps (int): steps to forecast.
        y_test (pd.Series): series of true future values.
        y_hist (pd.Series): series of historical values.
        offset_list (list[int]): offsets to start the forecast from in the test values. 
        linear_models (dict): dictionary of linear models.
        non_linear_models (dict): dictionary of non-linear models.
        naive (bool): whether to include naive forecast.
        time_step (str): time step of the time series (e.g. "h", "D")
    
    Returns:
        tuple[dict, dict, plt.Figure]: dictionary of forecast figures, dictionary of bar plot figures, average bar plot figure
    """

    # Initalise instances of the model_mae_scores class for each model
    # List of model_mae_scores instances
    model_mae_list = {}
    for name, value in linear_models.items():
        model_mae = ModelMAEScores(name)
        model_mae_list[name] = model_mae

    for name, value in non_linear_models.items():
        model_mae = ModelMAEScores(name)
        model_mae_list[name] = model_mae

    # If using the naive model add it to the model_mae_list
    if naive:
        model_mae = ModelMAEScores("Naive")
        model_mae_list["Naive"] = model_mae

    # We are going to store all of the offset plots in a dict and then return them all at once
    forecast_figs = {}
    bar_plot_figs = {}

    # Loop over offsets
    for offset in offset_list:
        # Create a copy of y_hist and y_test to avoid any changes to the original
        y_test_copy = copy.deepcopy(y_test)
        y_hist_copy = copy.deepcopy(y_hist)

        # Apply offset to y_hist_copy and y_test_copy 
        y_hist_copy = pd.concat([y_hist_copy, y_test_copy.iloc[:offset]])
        y_test_copy = y_test_copy.iloc[offset:]



        # Compute naive predictions
        # Today = yesterday
        y_pred_naive = y_test_copy.shift(1)
        y_pred_naive.iloc[0] = y_hist_copy.iloc[-1]


        # Store MAE scores for barplot and for the model_mae_list
        mae_scores = {}
            
        # Get real values
        y_real = y_test_copy.iloc[0:steps]
            
        # Plot 
        y_real_plot = to_NYC(y_real, time_step)
        if time_step == "h":
            ax = y_real_plot.plot(color='0.25', style='.', title=f"Forecast steps: {steps}, start date {y_real_plot.index[0]}, offset: {offset}")
        else: 
            ax = y_real_plot.plot(color='0.25', style='.', title=f"Forecast steps: {steps}, start date {str(y_real_plot.index[1].date())}, offset: {offset}")
        # Check if there are any linear models to forecast
        if len(linear_models) != 0:
            # Forecast the linear models:
            for name, value in linear_models.items():
                model = value[0]
                dp = value[1]
                hybrid = value[2]
                lags = value[3]
                    
                # Get forecast (we use cpu for linear forecasts, hence set gpu = False)
                y_fore_linear = forecast(model, y_hist_copy, lags, steps, offset, dp, hybrid, False)

                    
                # Compute MAE linear
                mae_linear = mean_absolute_error(y_fore_linear, y_real)
                mae_scores[name] = mae_linear

                # Only display for offsets in offsets_to_show 
                print(f"MAE: {mae_linear:.2f} for step = {steps}, model = {name}")

                # Add to plot
                # Convert to NYC time for plotting
                y_fore_linear = to_NYC(y_fore_linear, time_step)

                ax = y_fore_linear.plot(ax = ax, label = name)
            

        # check if there are any non linear models to forecast
        if len(non_linear_models) != 0:
            # Forecast the non linear models::
            for name, value in non_linear_models.items():
                model = value[0]
                dp = value[1]
                hybrid = value[2]
                lags = value[3]

                # Check whether using gpu
                if config["xgboost_setup"]["device"] == "cuda":
                    gpu = True
                else:
                    gpu = False

                # Get forecast
                y_fore_non_linear = forecast(model, y_hist_copy, lags, steps, offset, dp, hybrid, gpu)

                # Compute MAE non linear
                mae_non_linear = mean_absolute_error(y_fore_non_linear, y_real)
                mae_scores[name] = mae_non_linear

                print(f"MAE: {mae_non_linear:.2f} for step = {steps}, model = {name}")

                # Add to plot
                # Convert to NYC time for plotting
                y_fore_non_linear = to_NYC(y_fore_non_linear, time_step)
                       
                ax = y_fore_non_linear.plot(ax = ax, label = name)
        

            
        if naive == True:
            # Compute naive MAE
            y_step_pred_naive = y_pred_naive.loc[y_real.index]
            
            mae_naive = mean_absolute_error(y_real, y_step_pred_naive)
            mae_scores["Naive"] = mae_naive

            print(f"MAE: {mae_naive:.2f} for step =  30, model = Naive\n")

            # Plot forecasts
            # Convert to NYC time for plotting
            y_step_pred_naive = to_NYC(y_step_pred_naive, time_step)

            ax = y_step_pred_naive.plot(ax = ax, label = "Naive")
                
            
                
        # Save the mae scores to the model_mae_list
        model_mae_list = save_mae_scores(model_mae_list, mae_scores, steps, offset)
            
        # Add legend
        ax.legend()
        ax.set_ylabel("Trip counts")
        ax.set_xlabel("Pickup datetime")
        plt.xticks(rotation = 90, ha = "right")

        # Add to list of figures to return
        forecast_fig = ax.get_figure()
        forecast_figs[offset] = forecast_fig
        
        plt.close()

        # Plot MAE bar plots:
        df_mae = pd.DataFrame(list(mae_scores.items()), columns=["Model", "MAE"]) 

        plt.figure(figsize=(8,5))
        sns.barplot(data=df_mae, x="Model", y="MAE", hue = "Model")

        if time_step == "h":
            # Use the NYC time for title 
            plt.title(f"Model Comparison by MAE, steps = {steps}, start date = {y_real_plot.index[0]}, offset = {offset}")
        else:
            plt.title(f"Model Comparison by MAE, steps = {steps}, start date = {str(y_real_plot.index[0].date())}, offset = {offset}")
        plt.xticks(rotation=90, ha="right")

        barplot_fig = plt.gcf()
        bar_plot_figs[offset] = barplot_fig

        plt.close()

    # Now calculate the average MAE by step for each model and put into a dataframe
    df_avg_mae = create_avg_mae_df(model_mae_list, linear_models, non_linear_models, naive)  

    # Create bar plot of average MAE by step for each model for all steps
    avg_bar_plot_fig = create_avg_mae_barplot(df_avg_mae)

    return forecast_figs, bar_plot_figs, avg_bar_plot_fig
