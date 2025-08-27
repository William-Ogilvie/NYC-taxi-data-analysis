import pandas as pd
from pathlib import Path
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# We need to use the notebook version of tqdm if possible so it renders property in Jupyter Lab
if __name__ == "__main__":
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm



# src/jfk_taxis/ is location of current file so we go two above to get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" 
DATA_SAVE = PROJECT_ROOT / "data" / "processed"
DATA_SAVE_STRING = str(DATA_SAVE.resolve())

# Loads all parquets for a specific year (we do it one year at a time because each year already has approx 30 Million rows which is a lot for pandas to process)
def load_parquet(year: int) -> pd.DataFrame:
 
    # Get all files for that year
    files = DATA_DIR.glob(f"yellow*{str(year)}*.parquet")
   
    # Load and concatenate all the data into a single data frame (for tqdm leave = False ensures the bar disappears when done):
    df = pd.concat((pd.read_parquet(f, columns = ["tpep_pickup_datetime", "PULocationID"]) for f in tqdm(files, desc = "Download files: ", leave = False)), ignore_index = True)
    
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
        df = df.groupby('pickup_date').size().reset_index()
        df = df.rename(columns= {0: 'trips'})

        return df
    elif feature == "hour":
        df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour

        df = df.groupby(['pickup_date', 'pickup_hour']).size().reset_index()
        df = df.rename(columns= {0: 'trips'})

        # Convert the date and hour back into a datetime object
        df["day"] = pd.to_datetime(df["pickup_date"])
        df["dt"] = df["day"] + pd.to_timedelta(df["pickup_hour"], unit = "h")
        df = df.sort_values("dt")
        df = df.drop(["day"], axis = 1)
        df["dt"] = pd.to_datetime(df["dt"])



        return df
    else:
        print("Invalid feature entered for create_ts.")

# This function will load all the parquets, process them, create time series and save both the cleaned data and the time series to ../data/interim
def process_taxi_data(years: list[int], features: list[str]):
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
            ts.to_csv(DATA_SAVE_STRING + "/ts_" + feature + str(year) + ".csv", index = False)
        bar.update(1)

        # Save data
        bar.set_description("Save data") 
        df.to_parquet(DATA_SAVE_STRING + "/processed_" + str(year) + ".parquet")

        df_jfk.to_parquet(DATA_SAVE_STRING + "/processed_jfk_" + str(year) + ".parquet")
        bar.update(1) 

        bar.close()

# Function to loop through the list of years, produce a head and a basic forecast plot
def taxi_data_visuals(years: list[int]):

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

# Creates time series plots
def ts_plots(df: pd.DataFrame, feature: str, year: int, month: list[int]):
    sns.set_theme(style="darkgrid") 
    if feature == "daily":
        ax = sns.lineplot(data = df, x = "pickup_date", y = "trips")
        ax.set(title = f"JFK Airport yellow taxi trips per day - {year}", xlabel = "Date", ylabel = "Trips")

        # Show only one x axis tick per month
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 1))

        plt.xticks(rotation = 45, ha = "right")        
        plt.show()

    elif feature == "hour":
        # Create a subset using month:
        if len(month) != 0:
            df = df[df["dt"].dt.month.isin([month[0], month[1]])]
        

        ax = sns.lineplot(data = df, x = "dt", y = "trips")
        ax.set(title = f"JFK Airport hourly Yellow taxi trips - {year}", xlabel = "", ylabel = "Trips")
        
        plt.xticks(rotation = 45, ha = "right")
        plt.show()
    else:
        print("Invalid feature")

# Combines all the daily and hourly times series into two individual csvs
def combine_ts(years: list[int]):
    df_daily = pd.concat((pd.read_csv(DATA_SAVE_STRING + "/ts_daily" +  str(year) + ".csv") for year in years), ignore_index= True)      
    df_hour = pd.concat((pd.read_csv(DATA_SAVE_STRING + "/ts_hour" + str(year) + ".csv") for year in years), ignore_index= True)

    df_daily.to_csv(DATA_SAVE_STRING + f"/ts_daily{years[0]}-{years[-1]}.csv", index = False)
    df_hour.to_csv(DATA_SAVE_STRING + f"/ts_hour{years[0]}-{years[-1]}.csv", index = False)

# Plots the full daily and hourly ts
def plot_full_ts(df_daily: pd.DataFrame, years: list[int]):
    sns.set_theme(style="darkgrid")

    # Plot the daily ts
    ax = sns.lineplot(data = df_daily, x = "pickup_date", y = "trips")
    ax.set(title = f"JFK Airport yellow taxi trips per day - {years[0]}-{years[-1]}", xlabel = "Date", ylabel = "Trips")

    # Show only one x axis tick per month
    ax.xaxis.set_major_locator(mdates.YearLocator(base = 1))

    plt.xticks(rotation = 45, ha = "right")        
    plt.show()


def main():
    years = [2024]
    features = ["hour", "daily"]
    process_taxi_data(years, features)

if __name__ == "__main__":
    main()
