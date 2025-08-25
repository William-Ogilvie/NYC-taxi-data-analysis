import glob
import pandas as pd

# Loads all parquets for a specific year (we do it one year at a time because each year already has approx 30 Million rows which is a lot for pandas to process)
def load_parquet(year: int) -> pd.DataFrame:
 
    # Get all files for that year
    files = glob.glob(f"../data/raw/yellow*{str(year)}*.parquet")
        
    # Load and concatenate all the data into a single data frame:
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index = True)

    return df

# Does an inital clean of the data frame 
def init_clean_df(df: pd.DataFrame, year: int) -> pd.DataFrame:
    '''
    There are a couple things that can be cleaned almost immideately, the main one is that not all the data
    belongs to the specified year. A lot of these entries are on the overlap from New Years Eve to New Years Day of 
    previous years, so I suspect that its an error in keeping these entires. Potentially as they are being treated as special
    due to being in two different years. 
    '''


    df = df[df['tpep_pickup_datetime'].dt.year == year]

    return df

# Selects the JFK data
def select_jfk(df: pd.DataFrame) -> pd.DataFrame:
    '''
    From EDA we know that JFK airport has location ID 132
    Again we look at pickup taxi location rather than drop off 
    '''

    return df[df["PULocationID"] == 132].copy()


# Creates a time series for a specified feature to group by 
def create_ts(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    '''
    For now these ts will all be created from the 'tpep_pickup_datetime' column 
    The feature will be passed as a string:
    "hour" means an hourly breakdown
    "daily" means daily breakdown
    '''


    if feature == "daily":
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date

        return df.groupby('pickup_date').size().reset_index()
    elif feature == "hour":
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

        return df.groupby(['pickup_date', 'pickup_hour']).size().reset_index()
    else:
        print("Invalid feature entered for create_ts.")

# This function will load all the parquets, process them, create time series and save both the cleaned data and the time series to ../data/interim
def process_taxi_data(years: list[int], features: list[str]):
    '''
    years = years of taxi data to process, list of ints
    features = features to extract for time series, list of str 
    '''    

    # Loops through years and features creating both cleaned data frames and ts for features
    for year in years:

        df = load_parquet(year)
        df = init_clean_df(df, year)
        df_jfk = select_jfk(df)


        for feature in features:
            ts = create_ts(df_jfk, feature)
            ts.to_csv("../data/interim/ts_" + feature + year + ".csv")

        df.to_parquet("../data/interim/processed_" + str(year) + ".parquet")
        df_jfk.to_parquet("../data/interim/processed_jfk_" + str(year) + ".parquet")
            
