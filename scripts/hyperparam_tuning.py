"""
hyperparam_tuning
=================
This script runs the hyperparamter tuning from the notebooks using Optuna. Desgined to be run on an EC2 instance.
"""

import optuna
from jfk_taxis import load_config, create_val_data, objective_optuna, save_hyperparams, save_obj, load_ts_data, split_test_train_sets, make_offsets, save_config
from sklearn.linear_model import LinearRegression
from functools import partial

# --- Load config and constants ---
# Load config and project root
config, PROJECT_ROOT = load_config()

# First we assemble the design and model signatures from config.yml
DAILY_LINEAR_SIG = config["model_sigs"]["daily_linear"]
DAILY_HYBRID_SIG = config["model_sigs"]["daily_hybrid"]
HOURLY_LINEAR_SIG = config["model_sigs"]["hourly_linear"]
HOURLY_HYBRID_SIG = config["model_sigs"]["hourly_hybrid"]

# Now we get the model prefixes from config.yml
DAILY_LINEAR_PREFIX = config["model_naming"]["linear_model_prefix"] + "2"
DAILY_HYBRID_PREFIX = config["model_naming"]["hybrid_model_prefix"] + "2"
HOURLY_LINEAR_PREFIX = config["model_naming"]["linear_model_prefix"] + "2"
HOURLY_HYBRID_PREFIX = config["model_naming"]["hybrid_model_prefix"] + "2"
DEFAULT_NON_LINEAR_PREFIX = config["model_naming"]["default_non_linear"]

# Daily prefix
DAILY_PREFIX = config["shap"]["daily_prefix"]

# Hourly prefix 
HOURLY_PREFIX = config["shap"]["hourly_prefix"]

# Titles for plots and to key values for dicts
DAILY_LINEAR_NAME = f"{DAILY_PREFIX}_{DAILY_LINEAR_PREFIX}"
DAILY_HYBRID_NAME = f"{DAILY_PREFIX}_{DAILY_HYBRID_PREFIX}"
DAILY_NON_LINEAR_NAME = f"{DAILY_PREFIX}_{DEFAULT_NON_LINEAR_PREFIX}"
HOURLY_LINEAR_NAME = f"{HOURLY_PREFIX}_{HOURLY_LINEAR_PREFIX}"
HOURLY_HYBRID_NAME = f"{HOURLY_PREFIX}_{HOURLY_HYBRID_PREFIX}"
HOURLY_NON_LINEAR_NAME = f"{HOURLY_PREFIX}_{DEFAULT_NON_LINEAR_PREFIX}"

# --- Load data ---
# Load ts data
ts_daily, ts_hourly = load_ts_data()

# Create test train sets as defined in config.yml
ts_daily_train, ts_daily_test, ts_hourly_train, ts_hourly_test = split_test_train_sets(ts_daily, ts_hourly)

# --- Hyperparameter tuning ---
# We are going to need to create offsets for the test sets the same way we did in the modelling section
# Max time we can go up to in the test set to ensure we can still forecast
total_daily_time = config["hyperparameter_tuning"]["daily_test_size"] - config["hyperparameter_tuning"]["daily_steps"]

daily_offsets = make_offsets(total_daily_time, config["modelling"]["daily_offset_size"])

config["hyperparameter_tuning"]["daily_offset"] = daily_offsets

total_hourly_time = config["hyperparameter_tuning"]["hourly_test_size"] - config["hyperparameter_tuning"]["hourly_steps"]

hourly_offsets = make_offsets(total_hourly_time, config["modelling"]["hourly_offset_size"])

config["hyperparameter_tuning"]["hourly_offset"] = hourly_offsets

config, PROJECT_ROOT = save_config(config)

# Dictionary of parameters for Baysian optimisation
bayes_dict = {}

# Daily non_linear pre, incl and post COVID

# Set the parameters for creat_val_data and the objective function
n_splits = config["hyperparameter_tuning"]["n_splits"]
test_size = config["hyperparameter_tuning"]["daily_test_size"]
lags = config["shap"][DAILY_NON_LINEAR_NAME]["extracted_lags"]
constant = False
order = 0
fourier_features = config["shap"][DAILY_NON_LINEAR_NAME]["extracted_fourier_features"]
time_step = "D"
hybrid = None
steps = config["hyperparameter_tuning"]["daily_steps"]
offset_list = config["hyperparameter_tuning"]["daily_offset"]


# Looping through this dict avoids having to have three separate code cells, the keys are the sigs, values are the tsk
tmp_dict = {
    "daily_non_linear_pre_COVID": ts_daily_train[:"2020-01-01"],
    "daily_non_linear_incl_COVID": ts_daily_train,
    #"daily_non_linear_post_COVID": ts_daily_train["2022-01-01":] # We can't include post COVID as there is too little data to generate yearly lags
}

# Add to Bayes dict
for key, value in tmp_dict.items():
    bayes_dict[key] = {
        "n_splits" : n_splits,
        "test_size" : test_size,
        "lags" : lags,
        "constant" : constant,
        "order" : order,
        "fourier_features" : fourier_features,
        "time_step" : time_step,
        "ts" : value,
        "hybrid" : hybrid,
        "steps" : steps,
        "offset_list" : offset_list
    }


# Daily non_linear hybrid pre, incl and post COVID

# Set the parameters for creat_val_data and the objective function
n_splits = config["hyperparameter_tuning"]["n_splits"]
test_size = config["hyperparameter_tuning"]["daily_test_size"]
lags = config["shap"][DAILY_HYBRID_NAME]["extracted_lags"]
constant = True
order = config["shap"]["daily_hybrid_order"] # As outlined above we use order 3 
fourier_features = config["shap"][DAILY_HYBRID_NAME]["extracted_fourier_features"]
time_step = "D"
hybrid = LinearRegression(fit_intercept= False)
steps = config["hyperparameter_tuning"]["daily_steps"]
offset_list = config["hyperparameter_tuning"]["daily_offset"]


# Looping through this dict avoids having to have three separate code cells, the keys are the sigs, values are the tsk
tmp_dict = {
    "daily_hybrid_non_linear_pre_COVID": ts_daily_train[:"2020-01-01"],
    "daily_hybrid_non_linear_incl_COVID": ts_daily_train,
    #"daily_hybrid_non_linear_post_COVID": ts_daily_train["2022-01-01":]
}

# Add to Bayes dict
for key, value in tmp_dict.items():
    bayes_dict[key] = {
        "n_splits" : n_splits,
        "test_size" : test_size,
        "lags" : lags,
        "constant" : constant,
        "order" : order,
        "fourier_features" : fourier_features,
        "time_step" : time_step,
        "ts" : value,
        "hybrid" : hybrid,
        "steps" : steps,
        "offset_list" : offset_list
    }

# Hourly non_linear pre, incl and post COVID

# Set the parameters for creat_val_data and the objective function
n_splits = config["hyperparameter_tuning"]["n_splits"]
test_size = config["hyperparameter_tuning"]["hourly_test_size"]
lags = config["shap"][HOURLY_NON_LINEAR_NAME]["extracted_lags"]
constant = False
order = 0
fourier_features = config["shap"][HOURLY_NON_LINEAR_NAME]["extracted_fourier_features"]
time_step = "h"
hybrid = None
steps = config["hyperparameter_tuning"]["hourly_steps"]
offset_list = config["hyperparameter_tuning"]["hourly_offset"]


# Looping through this dict avoids having to have three separate code cells, the keys are the sigs, values are the tsk
tmp_dict = {
    "hourly_non_linear_pre_COVID": ts_hourly_train[:"2020-01-01"],
    "hourly_non_linear_incl_COVID": ts_hourly_train,
    #"hourly_non_linear_post_COVID": ts_hourly_train["2022-01-01":]
}

# Add to Bayes dict
for key, value in tmp_dict.items():
    bayes_dict[key] = {
        "n_splits" : n_splits,
        "test_size" : test_size,
        "lags" : lags,
        "constant" : constant,
        "order" : order,
        "fourier_features" : fourier_features,
        "time_step" : time_step,
        "ts" : value,
        "hybrid" : hybrid,
        "steps" : steps,
        "offset_list" : offset_list
    }

# Hourly non_linear hybrid pre, incl and post COVID

# Set the parameters for creat_val_data and the objective function
n_splits = config["hyperparameter_tuning"]["n_splits"]
test_size = config["hyperparameter_tuning"]["hourly_test_size"]
lags = config["shap"][HOURLY_HYBRID_NAME]["extracted_lags"]
constant = True
order = config["shap"]["hourly_hybrid_order"] # As outlined above we use order 0
fourier_features = config["shap"][HOURLY_HYBRID_NAME]["extracted_fourier_features"]
time_step = "h"
hybrid = LinearRegression(fit_intercept= False)
steps = config["hyperparameter_tuning"]["hourly_steps"]
offset_list = config["hyperparameter_tuning"]["hourly_offset"]


# Looping through this dict avoids having to have three separate code cells, the keys are the sigs, values are the tsk
tmp_dict = {
    "hourly_hybrid_non_linear_pre_COVID": ts_hourly_train[:"2020-01-01"],
    "hourly_hybrid_non_linear_incl_COVID": ts_hourly_train,
    #"hourly_hybrid_non_linear_post_COVID": ts_hourly_train["2022-01-01":]
}

# Add to Bayes dict
for key, value in tmp_dict.items():
    bayes_dict[key] = {
        "n_splits" : n_splits,
        "test_size" : test_size,
        "lags" : lags,
        "constant" : constant,
        "order" : order,
        "fourier_features" : fourier_features,
        "time_step" : time_step,
        "ts" : value,
        "hybrid" : hybrid,
        "steps" : steps,
        "offset_list" : offset_list
    }


# Loop through the Bayes dict, create the folds and run fmin for 100 evals to optimise hyperparamters, save the hyperparameters and sigs to a .pkl object

# list to store the keys
sig_list = []

for key, value in bayes_dict.items():
    print(f"Running hyperparameter optimisation for {key}")
    
    # Create the folds
    fold_dict = create_val_data(value["n_splits"], value["test_size"], value["lags"], value["constant"], value["order"], value["fourier_features"], value["time_step"], value["ts"])

    # Set parameters of objective
    steps = value["steps"]
    hybrid = value["hybrid"]
    offset_list = value["offset_list"]

    # Create a new study
    study = optuna.create_study(direction = "minimize", study_name = "hyper_parameter_optimisation")
    study.optimize(partial(objective_optuna, fold_dict = fold_dict, steps = steps, hybrid = hybrid, offset_list = offset_list), n_trials = config["hyperparameter_tuning"]["max_evals"])

    print("Number of finisehed trials: ", len(study.trials))
    print("Best trial: ", study.best_trial.params)

    print(f" Value: {study.best_trial.value}")
    print(" Params: ")
    for key2, value2 in study.best_trial.params.items():
        print(f"    {key2}: {value2}")
    
    # Get best hyperparams and add the missing ones
    best_hyperparams = study.best_trial.params

    best_hyperparams["random_state"] = config["xgboost_setup"]["random_state"]
    best_hyperparams["eval_metric"] = config["xgboost_setup"]["eval_metric"]
    best_hyperparams["tree_method"] = config["xgboost_setup"]["tree_method"]
    best_hyperparams["device"] = config["xgboost_setup"]["device"]

    # Save hyperparams
    save_hyperparams(best_hyperparams, key)

    # Append the sig to the list
    sig_list.append(key)

        

    
    # # Log hyperparams 
    # print("The best hyperparamters are: ", "\n")
    # print(best_hyperparams)

    # # Before we save the hyperparams we need to change the type of max_alpha to int 
    # # as well as add some of the parameters that aren't in best_hyperparmas, n_estimators, learnining_rate, random_state, eval_metric, tree_method and device
    # # At the same time we may as well convert the remaining hyperparams to floats rather than np.float64 
    # for key2, value2 in best_hyperparams.items():
    #     if key2 != "max_depth":
    #         best_hyperparams[key2] = float(value2)
    #     else:
    #         best_hyperparams[key2] = int(value2)

    # best_hyperparams["n_estimators"] = space["n_estimators"]
    # best_hyperparams["learning_rate"] = space["learning_rate"]
    # best_hyperparams["random_state"] = space["random_state"]
    # best_hyperparams["eval_metric"] = space["eval_metric"]
    # best_hyperparams["tree_method"] = space["tree_method"]
    # best_hyperparams["device"] = space["device"]

    # # Save hyperparams
    # save_hyperparams(best_hyperparams, key)

    # # Append the sig to the list
    # sig_list.append(key)

# Save signatures
save_obj(sig_list, "hyperparam_sigs")