"""
data_processing
=================

This module contains functions for processing the NYC taxi data, including loading, cleaning, filtering for JFK airport,
creating time series, and saving processed data. It also includes functions for visualizing the data through various plots.

The main function is process_taxi_data which orchestrates the entire data processing workflow.
"""



# --- Imports ---
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from .loading_helpers import load_config

if __name__ == "__main__":
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

# --- Load config ---
config, PROJECT_ROOT = load_config()    


# --- Constants and Paths ---
# Data directories
DATA_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"] 
DATA_SAVE = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["processed_path"]

# Saving conventions
ORIGINAL_PARQUET_PREFIX = config["saving"]["original_parquet_prefix"]
JFK_PARQUET_PREFIX = config["saving"]["jfk_parquet_prefix"]
TS_PREFIX = config["saving"]["ts_prefix"]
TS_DAILY = config["saving"]["ts_daily"] 
TS_HOURLY = config["saving"]["ts_hourly"]


# --- Functions ---
def load_parquet(year: int) -> pd.DataFrame:
    """ Loads all parquet files for a specific year and concatenates them into a single dataframe.

    Args:
        year (int): the year for which to load the parquet files.

    Returns:
        pd.DataFrame: a dataframe containing all the data for the specified year.
    """    
 
    # Get all files for that year
    files = DATA_DIR.glob(f"yellow*{str(year)}*.parquet")
   
    # Load and concatenate all the data into a single data frame (for tqdm leave = False ensures the bar disappears when done):
    df = pd.concat((pd.read_parquet(f, columns = ["tpep_pickup_datetime", "PULocationID"]) for f in tqdm(files, desc = "Download files: ", leave = False)), ignore_index = True)
    
    return df

def init_clean_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """ Initial cleaning of the dataframe.

    Args:
        df (pd.DataFrame): the dataframe to clean.
        year (int): the year for which to clean the dataframe.

    Returns:
        pd.DataFrame: the cleaned dataframe.
    """    


    '''
    There are a couple things that can be cleaned almost immideately, the main one is that not all the data
    belongs to the specified year. A lot of these entries are on the overlap from New Years Eve to New Years Day of 
    previous years, so I suspect that its an error in keeping these entires. Potentially as they are being treated as special
    due to being in two different years. 
    '''

    
    df = df[df['tpep_pickup_datetime'].dt.year == year]

    return df

def select_jfk(df: pd.DataFrame) -> pd.DataFrame:
    """ Selects the JFK airport taxi data from the dataframe.

    Args:
        df (pd.DataFrame): the dataframe to filter.

    Returns:
        pd.DataFrame: the dataframe containing only JFK airport taxi data.
    """    


    '''
    From EDA we know that JFK airport has location ID 132
    Again we look at pickup taxi location rather than drop off 
    '''

    return df[df["PULocationID"] == 132].copy()

def to_utc_hourly(s: pd.Series, source_tz: str = "America/New_York") -> pd.Series:
    """Convert a pandas Series to UTC hourly.

    Args:
        s (pd.Series): the input time series.
        source_tz (str, optional): the source timezone. Defaults to "America/New_York".

    Returns:
        pd.Series: the converted time series.
    """     
    s = s.copy()

    # Convert to datetime if not already
    s = pd.to_datetime(s)

    # Localise to source timezone (NYC)
    s = s.dt.tz_localize(
        source_tz, 
        ambiguous= "infer",           # handles the duplicated hour on fall-back, we use ambiguous=False to take the earlier (DST) hour
        nonexistent="shift_forward" # handles the missing hour on spring-forward by shifting the non existent time forward to nearest valid time
    )
    
    # Convert to UTC for modeling (no DST problems)
    s = s.dt.tz_convert("UTC")

    return s

def convert_to_NYC(s: pd.Series) -> pd.Series:
    """ Convert a pandas Series to America/New_York timezone.

    Args:
        s (pd.Series): the input time series.

    Returns:
        pd.Series: the converted time series.
    """     
    s = s.copy()

    # Convert to datetime if not already
    s = pd.to_datetime(s)

    # Convert to NYC timezone
    s.index = s.index.tz_convert("America/New_York")

    return s

"""
Cute demo from AI to show what we are doing/why:

import pandas as pd

# Summer date (EDT)
s = pd.DatetimeIndex(['2025-07-10 19:00'])
nyc = s.tz_localize('America/New_York')         # 2025-07-10 19:00-04:00
utc = nyc.tz_convert('UTC')                     # 2025-07-10 23:00+00:00
back = utc.tz_convert('America/New_York')       # 2025-07-10 19:00-04:00

# Winter date (EST)
s2 = pd.DatetimeIndex(['2025-01-10 19:00'])
nyc2 = s2.tz_localize('America/New_York')       # 2025-01-10 19:00-05:00
utc2 = nyc2.tz_convert('UTC')                   # 2025-01-11 00:00+00:00
back2 = utc2.tz_convert('America/New_York')     # 2025-01-10 19:00-05:00
"""

def create_ts(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """ Creates a time series dataframe for the specified feature.

    Args:
        df (pd.DataFrame): the dataframe to create the time series from.
        feature (str): the feature to create the time series for.

    Returns:
        pd.DataFrame: the time series dataframe.
    """

    '''
    For now these ts will all be created from the 'tpep_pickup_datetime' column 
    The feature will be passed as a string:
    "hour" means an hourly breakdown
    "daily" means daily breakdown
    '''

    # Get the pickup column
    df_time = df["tpep_pickup_datetime"].copy()

    # Convert to datetime
    df_time = pd.to_datetime(df_time)

    # Convert to UTC
    df_time = to_utc_hourly(df_time, source_tz="America/New_York")

    # Set this as the time series index
    df_time.index = df_time

    if feature == "daily":
        # Resample to daily frequency
        df_daily = pd.Series(1, index=df_time).resample("D").sum()
        df_daily.index.name = "pickup_date"  # Rename index for saving
        df_daily.name = "trips"            # Rename column for saving

        # Fill in missing days
        df_daily = df_daily.asfreq("D", fill_value= 0)

        return df_daily
    elif feature == "hour":
        # Resample to hourly frequency
        df_hourly = pd.Series(1, index = df_time).resample("h").sum()
        df_hourly.index.name = "dt"           # Rename index for saving
        df_hourly.name = "trips"            # Rename column for saving

        # Fill in missing hours
        df_hourly = df_hourly.asfreq("h", fill_value= 0)

        return df_hourly
    else:
        print("Invalid feature entered for create_ts.")

def process_taxi_data(years: list[int], features: list[str]) -> None:
    """ Processes the taxi data for the specified years and features.

    Args:
        years (list[int]): the years of taxi data to process.
        features (list[str]): the features to extract for time series.
    """    


    '''
    years = years of taxi data to process, list of ints
    features = features to extract for time series, list of str 
    '''    

    # Loops through years and features creating both cleaned data frames and ts for features
    for year in tqdm(years, desc = "Processing years"):

        # tqdm bar for each of the processing steps
        bar = tqdm(total = 5)


        # Load data
        bar.set_description("Loading data")
        df = load_parquet(year)
        bar.update(1)

        # Clean data
        bar.set_description("Initial clean")
        df = init_clean_df(df, year)
        bar.update(1)

        # Select JFK
        bar.set_description("Select JFK")
        df_jfk = select_jfk(df)
        bar.update(1)

        # Create ts
        bar.set_description("Create time series")
        for feature in features:
            ts = create_ts(df_jfk, feature)
            csv_path = DATA_SAVE / f"{TS_PREFIX}_{feature}{year}.csv"
            ts.to_csv(csv_path, index = True)
        bar.update(1)

        # Save data
        bar.set_description("Save data")
        parquet_original_path = DATA_SAVE / f"{ORIGINAL_PARQUET_PREFIX}_{year}.parquet"
        df.to_parquet(parquet_original_path)

        parquet_jfk_path = DATA_SAVE / f"{JFK_PARQUET_PREFIX}_{year}.parquet"
        df_jfk.to_parquet(parquet_jfk_path)
        bar.update(1) 

        bar.close()

def taxi_data_visuals(years: list[int]) -> None:
    """ Generates visualizations for taxi data (head and basic forecast plot).

    Args:
        years (list[int]): the years of taxi data to visualize.
    """    

    for year in tqdm(years, desc= "Year"):
       
        # Load data 
        df = load_parquet(year)

        # Visualise data
        display(df.head())
        print("Shape:", df.shape)
        display(df.isna().sum().to_frame("nulls"))
        
        # Daily time series
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date

        # Trips per day
        df_daily_counts = df.groupby('pickup_date').size()

        # Plot
        ax = df_daily_counts.plot(figsize = (12, 6), title=f"Trips per day - {year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Trips")
        plt.tight_layout()
        plt.show()

def ts_plots(df: pd.DataFrame, feature: str, year: int, month: list[int]) -> None:
    """ Creates time series plots for the specified feature and year.

    Args:
        df (pd.DataFrame): the data frame containing the taxi data.
        feature (str): the feature to plot (e.g., "daily" or "hourly").
        year (int): the year of the data.
        month (list[int]): month split to filter on, e.g. [1, 2] for January and February, if None just plot all
    """    

    sns.set_theme(style="darkgrid") 
    if feature == "daily":
        # Convert series index to NYC timezone 
        df = convert_to_NYC(df)

        ax = sns.lineplot(data = df, x = "pickup_date", y = "trips")
        ax.set(title = f"JFK Airport yellow taxi trips per day - {year}", xlabel = "Date", ylabel = "Trips")

        # Show only one x axis tick per month
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 1))

        plt.xticks(rotation = 45, ha = "right")        
        plt.show()

    elif feature == "hour":
        # Convert df["dt"] to datetime if not already
        df["dt"] = pd.to_datetime(df["dt"])

        # Create a subset using month:
        if len(month) != 0:
            df = df[df["dt"].dt.month.isin([month[0], month[1]])]
        

        ax = sns.lineplot(data = df, x = "dt", y = "trips")
        ax.set(title = f"JFK Airport hourly Yellow taxi trips - {year}", xlabel = "", ylabel = "Trips")
        
        plt.xticks(rotation = 45, ha = "right")
        plt.show()
    else:
        print("Invalid feature")

def combine_ts(years: list[int]) -> None:
    """ Combines all the daily and hourly time series into two individual CSV files.

    Args:
        years (list[int]): the years of taxi data to combine.
    """    


    df_daily = pd.concat((pd.read_csv(DATA_SAVE /  f"{TS_PREFIX}_{TS_DAILY}{year}.csv") for year in years), ignore_index= True)      
    df_hour = pd.concat((pd.read_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{year}.csv") for year in years), ignore_index= True)

    df_daily.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}{years[0]}-{years[-1]}.csv", index = False)
    df_hour.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{years[0]}-{years[-1]}.csv", index = False)

def plot_full_ts(df_daily: pd.DataFrame, years: list[int]) -> None:
    """ Plots the full daily time series.

    Args:
        df_daily (pd.DataFrame): the data frame containing the daily time series.
        years (list[int]): the years of the data.
    """    

    sns.set_theme(style="darkgrid")

    # Plot the daily ts
    ax = sns.lineplot(data = df_daily, x = "pickup_date", y = "trips")
    ax.set(title = f"JFK Airport yellow taxi trips per day - {years[0]}-{years[-1]}", xlabel = "Date", ylabel = "Trips")

    # Show only one x axis tick per month
    ax.xaxis.set_major_locator(mdates.YearLocator(base = 1))

    plt.xticks(rotation = 45, ha = "right")        
    plt.show()


# --- Testing ---
def main():
    """ Main function for testing purposes
    """    

    years = [2024]
    features = ["hour", "daily"]
    process_taxi_data(years, features)

if __name__ == "__main__":
    main()
