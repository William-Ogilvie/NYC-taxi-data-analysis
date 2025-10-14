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

    # As these are slightly different ways these files are formatted we will need to order them by month so when we concatonate we don't do it in the wrong order
    files = sorted(files, key = lambda x: int(x.stem.split("-")[-1])) # x.stem is the file name without the suffix, so when we split on "-" the month will be the last one in the list
   
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
        ambiguous= False,           # handles the duplicated hour on fall-back, we use ambiguous=False to take the earlier (DST) hour
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
    s.index = pd.to_datetime(s.index)

    # Convert to NYC timezone
    s = s.tz_convert("America/New_York")

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
    """ Generates visualizations for taxi data (head and basic forecast plot). This function has to be optimised to avoid crashing the docker container by using too much memory

    Args:
        years (list[int]): the years of taxi data to visualize.
    """    
    import gc

    for year in tqdm(years, desc= "Year"):
        # Get all files for that year
        files = DATA_DIR.glob(f"yellow*{str(year)}*.parquet")

        # As these are slightly different ways these files are formatted we will need to order them by month so when we concatonate we don't do it in the wrong order
        files = sorted(files, key = lambda x: int(x.stem.split("-")[-1])) # x.stem is the file name without the suffix, so when we split on "-" the month will be the last one in the list

        # Load just first file for display
        df_first = pd.read_parquet(files[0], columns = ["tpep_pickup_datetime", "PULocationID"])
        display(df_first.head())
        del df_first

        # Process files incrementally for counts
        daily_counts = {}
        total_rows = 0

        for f in tqdm(files, desc="Processing files", leave = False):
            df_chunk = pd.read_parquet(f, columns = ["tpep_pickup_datetime", "PULocationID"])
            total_rows += len(df_chunk)

            # Get dates and count
            dates = df_chunk["tpep_pickup_datetime"].dt.date
            for date, count in dates.value_counts().items():
                daily_counts[date] = daily_counts.get(date, 0) + count
            
            del df_chunk
            del dates
            gc.collect()
        
        print(f"Shape: ({total_rows}, 2)")

        # Convert to Series for plotting
        df_daily_counts = pd.Series(daily_counts).sort_index()

        # Plot
        ax = df_daily_counts.plot(figsize = (12, 6), title=f"Trips per day - {year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Trips")
        plt.tight_layout()
        plt.show()

        # Clean up
        del df_daily_counts
        del daily_counts
        gc.collect()
       

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
        # Create a series with the daily trips and convert to NYC timezone 
        plot_series = pd.Series(df["trips"].values, index = df["pickup_date"])
        plot_series.index = pd.to_datetime(plot_series.index)  # Ensure datetime format
        plot_series = convert_to_NYC(plot_series)

        ax = sns.lineplot(data = plot_series)
        ax.set(title = f"JFK Airport yellow taxi trips per day - {year}", xlabel = "Date", ylabel = "Trips")

        # Show only one x axis tick per month
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 1))

        plt.xticks(rotation = 45, ha = "right")  
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / f"images/JFK_Airport_yellow_taxi_trips_per_day_{year}.svg")      
        plt.show()

    elif feature == "hour":
        # Create a series with the hourly trips and convert to NYC timezone
        plot_series = pd.Series(df["trips"].values, index = df["dt"])
        plot_series.index = pd.to_datetime(plot_series.index)  # Ensure datetime format
        plot_series = convert_to_NYC(plot_series)


        # Create a subset using month:
        if len(month) != 0:
            plot_series = plot_series[plot_series.index.month.isin([month[0], month[1]])]
            month_disp = f"{month[0]} to {month[1]}"
            month_disp_save = f"{month[0]}_to_{month[1]}"
        else:
            month_disp = ""
            month_disp_save = ""
            
        

        ax = sns.lineplot(data = plot_series)
        ax.set(title = f"JFK Airport hourly Yellow taxi trips - {year} {month_disp}", xlabel = "", ylabel = "Trips")
        
        plt.xticks(rotation = 45, ha = "right")
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / f"images/JFK_Airport_hourly_yellow_taxi_trips_{year}_{month_disp_save}.svg")
        plt.show()
    else:
        print("Invalid feature")

def combine_ts(years: list[int]) -> None:
    """ Combines all the daily and hourly time series into two individual CSV files.

    Args:
        years (list[int]): the years of taxi data to combine.
    """    


    df_daily = pd.concat((pd.read_csv(DATA_SAVE /  f"{TS_PREFIX}_{TS_DAILY}{year}.csv") for year in years), ignore_index= False)      
    df_hour = pd.concat((pd.read_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{year}.csv") for year in years), ignore_index= False)

    # Due to timezone conversions we have a slight overlap in the series, for example 2012-01-01 appears both in the daily 2011 and 2012 time series.
    # We need to add these duplicates together
    df_daily = df_daily.groupby("pickup_date").sum()
    df_hour = df_hour.groupby("dt").sum() 

    df_daily.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}{years[0]}-{years[-1]}.csv", index = True)
    df_hour.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{years[0]}-{years[-1]}.csv", index = True)

def plot_full_ts(df_daily: pd.DataFrame, years: list[int]) -> None:
    """ Plots the full daily time series.

    Args:
        df_daily (pd.DataFrame): the data frame containing the daily time series.
        years (list[int]): the years of the data.
    """    

    sns.set_theme(style="darkgrid")

    # Get just the date part and convert to datetime
    df_daily["pickup_date"] = pd.to_datetime(df_daily["pickup_date"]).dt.date

    # Plot the daily ts
    ax = sns.lineplot(data = df_daily, x = "pickup_date", y = "trips")
    ax.set(title = f"JFK Airport yellow taxi trips per day - {years[0]}-{years[-1]}", xlabel = "Date", ylabel = "Trips")

    # Show only one x axis tick per month
    ax.xaxis.set_major_locator(mdates.YearLocator(base = 1))

    plt.xticks(rotation = 45, ha = "right") 
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / f"images/JFK_Airport_yellow_taxi_trips_per_day_{years[0]}-{years[-1]}.svg")       
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
