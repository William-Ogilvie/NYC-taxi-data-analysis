# NYC and JFK Airport Taxi Data Analysis
Exploring and visualising New York City Taxi trip data (2011-2025). With modelling of daily and hourly taxi pick up counts for JFK Airport using linear regresssion, xgboost and linear regression boosted on the residuals.  

## Table of Contents
- [Objective](#objective)
- [Data](#data)
- [Project Overview](#project-overview)
  - [1_EDA.ipynb](#1_EDA.ipynb)
- [Usage](#usage)
- [Results / Key Findings](#results--key-findings) 


## Objective

Our objective was to do some basic exploration of NYC taxi data using choropleths. Then to explore fitting various models to two time series for the taxi pick up count at JFK Airport, one daily and one hourly.

## Data 

We use the Yellow Taxi data provided by the NYC Taxi and Limousine Commission (TLC). The datasets themselves are reasonably large parquet files that record information about every single taxi trip for each month and year. In this project we primarily focus on tpep_pickup_date_time which is the time the taxi meter was engaged, so we treat this as the time of the pick up. As well as PULocationID which is the taxi zone the pick up occurred in. We use just these two columns to construct our time series for modelling. There is a data dictionary for the parquet files here: [Yellow Taxi Data Dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf). For context NYC is divided into various taxi zones by the TLC. The taxi zones themselves are split into four boroughs: [Bronx](https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_bronx.jpg), [Brooklyn](https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_brooklyn.jpg), [Manhattan](https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_manhattan.jpg), [Queens](https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_queens.jpg) and [Staten Island](https://www.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_staten_island.jpg). TLC have provided maps for this zones that you can view by clicking on each of the previous boroughs. There is .shp file that contains geo spatial data for the taxi zones that we will use when creating our choropleths during our EDA. There is also a taxi zone lookup csv that allows you to map taxi zone location IDs (which are what is stored inside the Yellow Taxi data) to the zone's name, borough, and service_zone.

Data source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 
via [NYC Open Data](https://opendata.cityofnewyork.us/).  
© City of New York. Data made available under the NYC Open Data Terms of Use.

## Project Overview

The project is setup around the five notebooks you can find in the notebooks directory. I have written a python package inside the src directory called jfk_taxis that contains helper functions for the notebooks that do a lot of the heavy lifting behind the scenes. We will briefly explain roughly what each notebook does here. Then later on in the README comment on our results/findings.

### 1_EDA.ipynb

This notebook is focused on doing some inital EDA of taxi drop offs and pick ups in the entire of NYC. We create interactive folium choropleths across both the years and months from 2011-2025, and comment on a few things that appear to be going on within the data.

### 2_data_processing.ipynb

This notebook handles the processing of the raw parquet files into the daily and hourly time series of taxi pick ups at JFK Airport.

### 3_EDA_JFK.ipynb

This notebook does some intial EDA into the time series themselves. Mostly to try and work out what seasonality might be within the time series (yearly, weekly, daily etc) and also to find what lags would be a good idea to use as features. 

### 4_modelling.ipynb

This notebook does our inital modelling. We start with three strands of model: linear regression, XGBoost (referred to as the non_linear model in our notebooks) and a hybrid model which is linear regression with boosted residuals (so linear regression with XGBoost fitted to the residuals of the linear regression). We use the features found in the previous notebook and split the data into a training and test set. To assess the quality of the model on the test set we compute the MAE for multi step forecasts of different step length (so for daily step lengths of 7, 30, 60 and for hourly 24, 168 and 720). As well as starting at different offsets throughout the test series, i.e. different starting dates for the forecast. We then take the average MAE across all the offsets and use that as a score to compare the models on each different step length. 

### 5_model_selection.ipynb

In this notebook we first compute the SHAP values for some of the best models from the previous notebook. We use this to create a reduced feature set by ranking feature importance by mean absolute SHAP value. We then test that this reduced feature set captures enough of the original signal to be useful. Using the same model evaluation scheme as in the previous notebook. Then we perform Bayesian hyperparamter tuning with Optuna to tune all the XGBoost models. Finally we use the evaluation scheme as in the previous notebook to find the best model for the two time series cases: daily and hourly.

#### Notes on hyperparamter tuning

Hyperparamter tuning can take a very long time depending on your machine. I did two versions of hyperparamter tuning one with early stopping to deterimne n_estimators (number of trees for XGBoost) and one without. Add comment on which is better. I found I needed a reasonably large number of trials and at least in the without early stopping case just 100 trials wasn't enough. This means the tuning can take a long time so I have provided pretuned hyperparamters inside the results/tuned_hyperparams directory. If you would like to use them when you run 5_model_selection to avoid having to tune the hyperparameters yourself, simply copy the files into saved_objects and run the notebook skipping the hyperparamter tuning steps (there are instructions in the notebook on how to do this).

## Usage

Clone the repository:
```bash
git clone https://github.com/William-Ogilvie/NYC-taxi-data-analysis.git
cd NYC-taxi-data-analysis
```

There is a Dockerfile inside the repository that you can build an image from. It has a micromamba base that will have CUDA installed by default on Ubuntu. There are then two enviroment YAML files that you can choose to build from, environment_cpu.yml and environment_gpu.yml. The GPU version will install the version of XGBoost with GPU support, the CPU version just runs on the CPU. On my machine there was a noticable improvement in fit times when using the GPU and the project is setup so that you can use either. To sepcifiy whether you want GPU or CPU whilst building the docker image use the --build-arg ENV_FILE=environment_gpu.yml/enviornment_gpu.yml.

The full docker commands are below:

```bash
docker build -t jfk-taxi:gpu --build-arg ENV_FILE=environment_gpu.yml .
```
or
```bash
docker build -t jfk-taxi:cpu --build-arg ENV_FILE=environemnt_cpu.yml .
```

To then run a container you will want to bind mount the current working directory to the container. This is because you want to be able to download the data as well as save and load .pkl files later on in the project. The project consits of several notebooks so we will setup the container so that you can access jupyter lab, by mapping ports 8888 to each other from the container and host. The project also has a streamlit app that demos some of the EDA and model building, so we will map ports 8501 to each other on the container and local host. If you are using your GPU for training you will need to also give the container access to it. It is also worth noting that some of the project can be slightly memory intensive so it is worth ensuring your container will have access to at least 8 GBs of RAM.

So the full docker run command is as follows:
```bash
docker run -it --rm --gpus all -p 8888:8888 -p 8501:8501 --mount type=bind,src="$(pwd)",dst=/app jfk-taxi:gpu
```
or
```bash
docker run -it --rm -p 8888:8888 -p 8501:8501 --mount type=bind,src="$(pwd)",dst=/app jfk-taxi:cpu
```

Now we will need to do some intial setup before we can run the notebooks. First we will need to install the package jfk_taxis which is located inside src/jfk_taixs. This package contains helper functions that do most of the heavy lifting and make our notebooks more readable. It also has unittests for all modules that you cand find in the tests dir. To install the package use the following command whilst in the \app dir:

```bash
pip install -e . 
```

Then we need to setup the config file to account for whether or not we should use GPU during training. It will also create all necessary directories for the project if they do not exist already. Go to the scripts directory and run setup.py. This performs a small test to determine whether XGBoost has been installed with CUDA support and updates the config.yml file accordingly (which can be found in config/config.yml). So run the following commands:

```bash
cd scripts
python setup.py
```

Then we need to download the data. Within scripts there is a file called get_parquet.py. This scrapes https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page using BeautifulSoup to get the urls to download the parquet files containing the taxi data, the .shap file for the taxi zones and the taxi zone lookup csv. This will then be saved to parquet_files.txt. So first run this script:

```bash
python get_parquet.py
```

Then we have a small bash script called download_and_extract.sh that just loops through all the files in parquet_files.txt and downloads them. To make the bash script executable run:

```bash
chmod +x donwload_and_extract.sh
```

Then run the script with:

```bash
./download_and_extract.sh
```

Once the data has been downloaded you will want to start with the notebooks at least initally to process the taxi data. Before moving onto the streamlit app. To launch jupyter lab in the container run the follwing command from the project root (so mambauser@USER_NAME:/app):

```bash
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''
```

You will then be able to open jupyter lab on your host machine by visting http://127.0.0.1:8888/ in a browser. From here you will be able to see the projects notebooks inside the notebooks dir. The notebooks are numbered from 1 to 5 and will walk through and explain the project in order. There is a folder of rough notebooks that I have kept from intial exploration of the data for completness although they may have errors within them. Once you have completed notebook 2_data_processing.ipynb the time series data will now have been processed allowing you to launch the streamlit app. However it is recommened to complete notebooks 3 through 5 first as they will provide better context for the app itself. 

To then launch the streamlit app run the following commands:

```bash 
cd app
streamlit run home.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

You will then be able to view the app on the host machine by going to http://127.0.0.1:8501/ in a browser. The app home page will give a brief explanation of how the app works and hopefully from reading notebooks 1_EDA.ipynb, 4_modelling.ipynb and 5_model_selection.ipynb you can understand and follow what parts of the project it is demonstrating.

### Using AWS for hyperparamter tuning

One of the things I found on my machine was that the Bayesian hyperparamter tuning in 5_model_selection.ipynb took a long time for a large number of trials. So I used an AWS EC2 instance to run the tuning over the course of several days whilst I worked on other parts of the project. We will briefly explain how to do this here. 

First you will need to create an S3 bucket to store both the time series data but also the results of the hyperparameter tuning. I named the S3 bucket jfk-taxi-data-william-ogilvie. If you choose a different name you will need to manually alter the bash script hyperparam_tuning_bash.sh, change BUCKET to the name of your bucket. Inside the S3 bucket place the full time series data (ts_daily2011-2025.csv, ts_hourly2011-2025.csv) inside a data/time_series directory. 

You will need to create an IAM role with AmazonS3FullAcess policy if you do not have one already. Then create the EC2 instance, I decided to use the Deep Learning Base AMI with Single CUDA (Ubuntu 22.04) AMI as it comes with git, docker and the AWS CLI pre installed and configured. I only have access to CPU instances so wasn't able to test the GPU functionality on AWS, but this instance should allow you to make use of the GPU if you would like. Make sure you give it the IAM role with AmazonS3FullAcess policy. 

Now if you are going to use a GPU isntance you will need to modify the hyperparam_tuning_bash.sh script. Specifically change USE_GPU to be 1 rather than 0. Then change ENV to environment_gpu.yml rather than enviornment_cpu.yml. You may also want to change the name of the docker image under IMAGE for completness. 

Then once you connect to the instance run the following commands inside the home directory:

```bash
mkdir -p project logs
cd project 
git clone https://github.com/William-Ogilvie/NYC-taxi-data-analysis.git
```

We wil be using tmux to run the hyperparam_tuning_bash.sh script even when we disconnect from the instance. So we will need to install it:

```bash
sudo apt-get update && sudo apt-get install -y tmux
```

We will create a new tmux session:

```bash
tmux new -s hyperparam_tuning
```

Inside the tmux session we will then run the hyperparm_tuning_bash.sh script inside the scripts dir. This script will load the data from the S3 bucket. It will then build the docker image and run a container of this image. Inside this container it will install the jfk-taxis package, run scripts/setup.py and then run scripts/hyperparam_tuning.py. It will produce logs and save them into the logs directory on the instance as well as save the tuned hyperparameters into outputs/$RUN_ID in your S3 bucket.

Run the hyperparam_tuning_bash.sh script inside the tmux session:

```bash
cd NYC-taxi-data-analysis/scripts
bash hyperparam_tuning_bash.sh 2>&1
```

You can then download the tuned hyperparameters from the S3 bucket and place them inside the data/saved_objects folder on your local version of the repository. To see them plotted run the first four cells of the 5_model_selection.ipynb notebook. Then run all remaining cells from the following cell:

```python
# Load signatures
hyper_sig = load_obj("hyperparam_sigs")

# Dict of hyperparams
hyper_dict = {}

# Load the hyperparams for each model
for sig in hyper_sig:
    hyper_params = load_hyperparams(sig)
    hyper_dict[sig] = hyper_params# Load signatures
hyper_sig = load_obj("hyperparam_sigs")
```

### Testing

There are unittests for the custom package jfk_taxis. They are written for pytest if you would like to run them simply run the following commands:

```bash
cd tests
pytest
```

## Reults / Key Findings

We initally perform EDA for the entire of the NYC taxi data, primarily using choropleths. We then shift our focus to JFK Airport and create two time series that count the total number of taxis at the Airport, one daily, one hourly. We explore these time series to try and find relevant predictive features that we could use for models. Our intial exploration suggests that daily and weekly fourier features could be good for our hourly time series, and weekly and yearly fourier features for our daily time series. We also get back a considerable number of lags for both time series that cover at least one year into the past. 

Then we fit some inital models to these featuers. We test three strands of model linear regression, xgboost (usually reffered to as our non linear model in notebooks) and boosted linear regression on the residuals (so xgboost is fitted to the residuals of a linear regression, and added to its predictions). We will use a Naive baseline of just predicting the current value as the same as that one time step (day or hour) ago. The use of this baseline originally was from trial and error among others like 1 week ago. However it is supported via the SHAP values as most models have a very large mean SHAP value for lag_1. Suggesting that the previous time step has considerable predictive power. (comment on findings here).

Later we explore feature importance using SHAP values for all the models. We take the top 30 features by average SHAP value and see if they retain most of the predictive signal. It turns out for xgboost the reduced feature set is similarly effective. However for just pure linear regression it does seem to be considerably worse and potentially not worth using as the computational savings are smaller compared to with xgboost. 

Finally we perform bayesian hyperparameter tuning with optuna. I personally ran this in on AWS EC2 instance over a few days to do 1000 trials (maybe change this idk?) but the improvements seem to be minimal unfortunately (maybe?!).

We conclude with the best models for the two cases (add here). 



Note explain the install of XGBoost for gpus carefully because we assume that u run one

Comment on fact we expected fourier features to be important, turns out according to SHAP values actaully lags matter way way more.

Make sure to explain that in terms of hourly lags we can't use all of them so we take first 300 approx and throw in roughly half year and full year ones, with full year being significant.

Explain time zone conversions maybe? like converting to UTC to avoid DST errors

maybe comment on how offsets work? every 30 days with a bit of jitter, so in total 7 offsets

Comments on what we discovered in 1_EDA so like the change over years, the seasonality of taxis how that informs us to be interested in seasonality at JFK Airport.

Comments on 3_EDA_JFK how we found seasonlaity + lags

Comments on 4_modelling maybe the best models found + explanation of how we measure "best model"

Comments on 5_model_selection comments on the fact the reduced feature sets go hard + the hyperparamter tuning etc.


