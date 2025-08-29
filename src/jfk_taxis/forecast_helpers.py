# Series of helper functions for time series forecasting
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.deterministic import DeterministicProcess
import seaborn as sns
from IPython.display import display
from statsmodels.tsa.deterministic import CalendarFourier, CalendarSeasonality

# Preprocess the data for the models:

# Function creates and returns the design matrix with both lags and an underlying deterministic process, the time series, and the deterministic process itself
def preprocess(lags, constant, order, fourier_features, ts):
    y = ts

    # When forecasting we need the index to have a frequency, for us this is daily
    y.index = pd.date_range(start=y.index[0], periods=len(y), freq="D")

    fourier_list = []
    # Fourier features for seasonality
    for feature in fourier_features:
        if feature == "YE":
            fourier_list.append(CalendarFourier(freq = "YE", order = 10)) # Annual seasonality (10 harmonics)
        elif feature == "W":
            fourier_list.append(CalendarFourier(freq = "W", order = 5)) # Weekly seasonality (3 harmonics)
        elif feature == "D":
            fourier_list.append(CalendarFourier(freq = "D", order = 5)) # Daily seasonality (5 harmonics)
   
   
    dp = DeterministicProcess(
        index = y.index,
        constant = constant,   # Dummy feature for bias (y-intercept)
        order = order,         # Polynomial trend (degree 1 = linear)
        seasonal = False,    # Don't use seasonal dummies
        additional_terms = fourier_list, # Add in the Fourier terms and any other extra features
        drop = True,       # Drop first column to avoid collinearity
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

    return (X, y, dp)
        
    

# Fit models:

# Fits linear regression to the design matrix, without an intercept due to the use of deterministic processes already including one
def fit_linear(X,y):
    model = LinearRegression(fit_intercept = False)
    model.fit(X,y)

    return model

# Fits XGBoost to the design matrix and time series
def fit_non_linear(X, y):
    # XGBoost:
    model_xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=37
    )
    model_xgb.fit(X, y);
    return model_xgb


# Creates a forecast for a specifided number of steps
def forecast(model, y, lags, steps, dp, hybrid):
    """
    model  = trained linear regression
    y      = pandas Series with historical values
    lags   = list of lags used in training (e.g. [1,2,3])
    steps  = how many steps ahead to forecast
    dp     = the deterministic process used
    hybrid = if not None, the hybrid model to add to the linear model
    """
    
    preds = []
    y_hist = y.copy()

    # Create the deterministic features for the forecast
    X_future_det = dp.out_of_sample(steps = steps)

    for i in range(steps): 

        # Get the deterministic row
        x_next = X_future_det.iloc[i].copy()
        
        # Create the lags using historical data
        # for j in lags:
        #     x_next[f'y_lag_{j}'] = y_hist.iloc[-j]
        lag_dict = {f'y_lag_{j}': y_hist.iloc[-j] for j in lags}
        x_next = pd.concat([x_next, pd.Series(lag_dict)], axis = 0)

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
        y_hist = pd.concat([y_hist, new_point])

        # Add prediction to preds series
        preds.append(new_point)

    # Turn preds into a pandas series
    preds = pd.concat(preds)
    return preds

# Create forecasts, plot and compare to naive baseline, the key difference is we will now pass two dicts of linear and non linear models for ease of use
def test_forecasts_dicts(steps, y_test, y_hist, linear_models, non_linear_models, lags):
    """
    steps = array of the step lengths to forecast
    y_test = pd.Series of the true future values
    y_hist = pd.Series of historical values
    linear_models = dict of linear models
    non_linear_models = dict of non linear models
    lags = lags used in the models
    """

    # Compute naive predictions
    # Today = yesterday
    y_pred_naive = y_test.shift(1)
    y_pred_naive.iloc[0] = y_hist.iloc[-1]
    
   
    for step in steps:

        # Store MAE scores for barplot
        mae_scores = {}
        
        # Get real values
        y_real = y_test.iloc[0:step]
        
        # Plot
        ax = y_real.plot(color='0.25', style='.', title=f"Forecast steps: {step}")
        
        # Forecast the linear models:
        for name, value in linear_models.items():
            model = value[0]
            dp = value[1]
            hybrid = value[2]
            
            # Get forecast
            y_fore_linear = forecast(model, y_hist, lags, step, dp, hybrid)

            # Compute MAE linear
            mae_linear = mean_absolute_error(y_fore_linear, y_real)
            mae_scores[name] = mae_linear
            print(f"MAE Linear: {mae_linear:.2f} for step = {step}, model = {name}")

            # Add to plot
            ax = y_fore_linear.plot(ax = ax, label = name)
        

        
        # Forecast the non linear models:
        for name, value in non_linear_models.items():
            model = value[0]
            dp = value[1]
            hybrid = value[2]

            # Get forecast
            y_fore_non_linear = forecast(model, y_hist, lags, step, dp, hybrid)
            
            # Compute MAE non linear
            mae_non_linear = mean_absolute_error(y_fore_non_linear, y_real)
            mae_scores[name] = mae_non_linear
            print(f"MAE Non Linear: {mae_non_linear:.2f} for step = {step}, model = {name}")

            # Add to plot
            ax = y_fore_non_linear.plot(ax = ax, label = name)
       

        
       
        # Compute naive MAE
        y_step_pred_naive = y_pred_naive.loc[y_real.index]
         
        mae_naive = mean_absolute_error(y_real, y_step_pred_naive)
        mae_scores["Naive"] = mae_naive
        print(f"Naive MAE: MAE = {mae_naive:.2f}\n")

        # Plot forecasts
        ax = y_step_pred_naive.plot(ax = ax, label = "Naive")
        ax.legend()
        plt.xticks(rotation = 90)
        plt.show()

        # Plot MAE bar plots:
        df_mae = pd.DataFrame(list(mae_scores.items()), columns=["Model", "MAE"]) 

        plt.figure(figsize=(8,5))
        sns.barplot(data=df_mae, x="Model", y="MAE")

        plt.title(f"Model Comparison by MAE, steps = {step}")
        plt.xticks(rotation=45, ha="right")
        plt.show()
    
# Runs the forecasts in question, must be passed as pandas series
def run_forecasts(steps, lags, linear_models, non_linear_models, old_ts: pd.Series, new_ts: pd.Series):
    
    y_test = new_ts
    y_hist = old_ts
    y_hist.index = pd.date_range(start=y_hist.index[0], periods=len(y_hist), freq="D")
    
    test_forecasts_dicts(steps, y_test, y_hist, linear_models, non_linear_models, lags)
    return 0