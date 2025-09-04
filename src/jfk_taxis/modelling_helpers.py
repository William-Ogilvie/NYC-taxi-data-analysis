from .forecast_helpers import fit_non_linear, preprocess, fit_linear 
from .training_helpers import save_design, save_models

# Function will create and return dict of design, target and deterministic process for non linear model
def create_design_non_linear(lags, fourier_features, time_step, ts, name):
    """
    lags: list of lags
    fourier_features: list of fourier features for dp
    time_step: time step of series so "D" or "h"
    ts: time series itself
    name: name of the model for the dictionary 
    """ 
    
    # Create non linear design and traget matricies
    (X_non_linear, y_non_linear, dp_non_linear) = preprocess(lags, False, 0, fourier_features, time_step, ts)

    # For the preprocess function the parameters are: lags, constant, order, fourier features, time_step (for the target series), time series

    # Store non linear design matricies
    non_linear_design = {
        name: (X_non_linear, y_non_linear, dp_non_linear)
    }

    return non_linear_design

# Function returns dict of design, target and deterministic process under specified model name
def create_design_linear(lags, order, fourier_features, time_step, ts, name):
    """
    lags: list of lags
    order: order of trend in dp (deterministic process)
    fourier_features: list of fourier features for dp
    time_step: time step of series so "D" or "h"
    ts: time series itself
    name: name of the model for the dictionary
    """

    # Create X,y, dp_linear
    (X,y, dp_linear) = preprocess(lags, True, order, fourier_features, time_step, ts, name)
    
    # Save as dict 
    linear_design = {
        name: (X, y, dp_linear)
    }

    return linear_design 

# Function trains the non linear models on the designs in the dict and returns them
def train_non_linear_models(non_linear_design):
    """
    non_linear_design: dict containing design, target and deterministic process 
    """

    # Dict for storing non_linear_models
    non_linear_models = {}

    # Loop through design dict and fit non_linear_models
    for key, value in non_linear_design.items():
        non_linear_models[key] = (fit_non_linear(value[0], value[1]), value[2], None)

    return non_linear_models 

# Function trains the linear models on the designs in the dict and returns them
def train_linear_models(linear_design):
    """
    linear_design: dict of design, target and deteriministic process for each linear model 
    """

    # Dict for storing linear_models
    linear_models = {}

    # Loop through design dict and fit linear_models

    for key, value in linear_design.items():
        linear_models[key] = (fit_linear(value[0], value[1]), value[2], None)


# Function creates design, target, deterministic process, the model and fits the model. Returning two dicts one of designs one of models
def create_train_non_linear(names, lags, fourier_features, time_step, ts):
    """
    names: list of names of the non_linear_models 
    lags: list of lags
    fourier_features: list of fourier features
    time_step: time step of series so "D" or "h"
    ts: time series itself
    """

    # Dict of non_linear design, target, dp
    non_linear_design = {}

    # Dict of non_linear models themselves
    non_linear_models = {}

    # Loop through names creating design, target and dp. Fit the models as well
    for name in names:
        # Create design 
        design = create_design_non_linear(lags, fourier_features, time_step, ts, name)

        # Train model
        model = train_non_linear_models(design)

        # Store design and model 
        non_linear_design[name] = design[name]
        non_linear_models[name] = model
    
    return non_linear_design, non_linear_models

# Function creates design, target, deterministic process, the model and fits the model. Returning two dicts one of designs one of models
def create_train_linear(names, order_list, lags, fourier_features, time_step, ts):
    """
    names: list of names of the non_linear_models
    order_list: list of orders to fit 
    lags: list of lags
    fourier_features: list of fourier features
    time_step: time step of series so "D" or "h"
    ts: time series itself
    """

    # Dict of linear design, target, dp
    linear_design = {}

    # Dict of linear models themselves
    linear_models = {}

    # Loop through names creating design, target and dp. Fit the models as well
    for name, order in zip(names, order_list):
        # Create design 
        design = create_design_linear(lags, order, fourier_features, time_step, ts, name)

        # Train model
        model = train_linear_models(design)

        # Store design and model 
        linear_design[name] = design[name]
        linear_models[name] = model
    
    return linear_design, linear_models

# Function will create save and train non linear models and either linear models or hybrid models
def create_train_save_models(names_linear, names_non_linear, hybrid, sig, order_list, lags, fourier_features, time_step, ts):
    """
    names_linear: list of names for the linear models 
    names_non_linear: list of names for the non linear models
    hybrid: the hybrid model to be used
    sig: signature to name the pkl objects when saved (e.g. 5_order_linear_dalily)
    order_list: the list of orders for the linear trend
    lags: list of lags
    fourier features: list of fourier features
    time_step: time step to use either "D" or "h"
    ts: time series itself
    """

    # Dict of linear or hybrid designs
    linear_design = {}

    # Dict of linear or hybrid models
    linear_models = {}

    # Dict of non linear designs
    non_linear_design = {}

    # Dict of non_linear models
    non_linear_models = {}

    # First do the case of no hybrid models
    if hybrid is None:
        linear_design, linear_models = create_train_linear(names_linear, order_list, lags, fourier_features, time_step, ts)
    # TODO hybrid case

    # Create and train non linear models
    non_linear_design, non_linear_models = create_train_non_linear(names_non_linear, lags, fourier_features, time_step, ts)

    # Save designs and models
    save_design(linear_design, non_linear_design, sig)
    save_models(linear_models, non_linear_models, sig)
