"""
test_data_processing.py
=========================

Unit test for data_processing.py.
""" 

def test_load_parquet():
    """ test for load_parquet function in data_processing.py, note load_parquet loads parquet files for a given year.
    As our data runs from 2011 to 2025 to avoid messing with our actual data we will use taxi data in NYC from 1974 (that we create here)
    """    
    import pandas as pd
    import os
    from jfk_taxis import load_config
    from jfk_taxis import data_processing

    # Get config
    config, PROJECT_ROOT = load_config()

    # Constants
    DATA_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"] 

    # Create taxi data for 1487
    df_1974_01 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1974-01-01 14:28:00", 
            "1974-01-15 21:32:00",
            "1974-01-28 09:01:00",
            "1974-01-30 16:50:00"
        ]),
        "PULocationID": [1, 2, 3, 123],
        "DOLocationID": [4, 5, 6, 7]
    }) 

    df_1974_07 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1974-07-02 14:28:00", 
            "1974-07-12 23:50:00",
            "1974-07-26 00:10:00",
            "1974-07-30 15:30:00"
        ]),
        "PULocationID": [100, 32, 46, 123],
        "DOLocationID": [3, 55, 56, 9]
    })

    df_1974_11 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1974-11-05 18:00:00", 
            "1974-11-15 02:50:00",
            "1974-11-20 06:34:00",
            "1974-11-30 19:23:00"
        ]),
        "PULocationID": [10, 22, 33, 123],
        "DOLocationID": [44, 55, 66, 77]
    })

    # To make sure we don't take any other years
    df_1975_01 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1975-01-01 17:21:00", 
            "1975-01-15 23:55:00",
        ]),
        "PULocationID": [11, 2],
        "DOLocationID": [4, 9]
    })


    # Save files
    df_1974_01.to_parquet(DATA_DIR / "yellow_tripdata_1974-01.parquet", index=False)
    df_1974_07.to_parquet(DATA_DIR / "yellow_trip____HELLOOOOOOOO!_data_1974-07.parquet", index=False) # as some of the real files have slightly different formatting between yellow and the year we include that in our tests
    df_1974_11.to_parquet(DATA_DIR / "yellow_tripdata_1974-11.parquet", index=False)
    df_1975_01.to_parquet(DATA_DIR / "yellow_tripdata_1975-01.parquet", index=False)

    # Run load_parquet for 1974
    df_loaded = data_processing.load_parquet(1974)

    # Remove the files we created
    os.remove(DATA_DIR / "yellow_tripdata_1974-01.parquet")
    os.remove(DATA_DIR / "yellow_trip____HELLOOOOOOOO!_data_1974-07.parquet")
    os.remove(DATA_DIR / "yellow_tripdata_1974-11.parquet")
    os.remove(DATA_DIR / "yellow_tripdata_1975-01.parquet")


    # Basic checks
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape[0] == 12 # 4 rows in each of the three files for 1974
    assert df_loaded.shape[1] == 2 # we only extract these two columns to save memory
    assert all(df_loaded.columns == ["tpep_pickup_datetime", "PULocationID"])
    assert all(df_loaded["tpep_pickup_datetime"].dt.year == 1974) # all returns true if all elts in iterable are true
    assert df_loaded.iloc[0]["tpep_pickup_datetime"] == pd.Timestamp("1974-01-01 14:28:00")
    assert df_loaded.iloc[0]["PULocationID"] == 1
    assert df_loaded.iloc[1]["tpep_pickup_datetime"] == pd.Timestamp("1974-01-15 21:32:00")
    assert df_loaded.iloc[1]["PULocationID"] == 2
    assert df_loaded.iloc[2]["tpep_pickup_datetime"] == pd.Timestamp("1974-01-28 09:01:00")
    assert df_loaded.iloc[2]["PULocationID"] == 3
    assert df_loaded.iloc[3]["tpep_pickup_datetime"] == pd.Timestamp("1974-01-30 16:50:00")
    assert df_loaded.iloc[3]["PULocationID"] == 123
    assert df_loaded.iloc[4]["tpep_pickup_datetime"] == pd.Timestamp("1974-07-02 14:28:00")
    assert df_loaded.iloc[4]["PULocationID"] == 100
    assert df_loaded.iloc[5]["tpep_pickup_datetime"] == pd.Timestamp("1974-07-12 23:50:00")
    assert df_loaded.iloc[5]["PULocationID"] == 32
    assert df_loaded.iloc[6]["tpep_pickup_datetime"] == pd.Timestamp("1974-07-26 00:10:00")
    assert df_loaded.iloc[6]["PULocationID"] == 46
    assert df_loaded.iloc[7]["tpep_pickup_datetime"] == pd.Timestamp("1974-07-30 15:30:00")
    assert df_loaded.iloc[7]["PULocationID"] == 123
    assert df_loaded.iloc[8]["tpep_pickup_datetime"] == pd.Timestamp("1974-11-05 18:00:00")
    assert df_loaded.iloc[8]["PULocationID"] == 10
    assert df_loaded.iloc[9]["tpep_pickup_datetime"] == pd.Timestamp("1974-11-15 02:50:00")
    assert df_loaded.iloc[9]["PULocationID"] == 22
    assert df_loaded.iloc[10]["tpep_pickup_datetime"] == pd.Timestamp("1974-11-20 06:34:00")
    assert df_loaded.iloc[10]["PULocationID"] == 33
    assert df_loaded.iloc[11]["tpep_pickup_datetime"] == pd.Timestamp("1974-11-30 19:23:00")
    assert df_loaded.iloc[11]["PULocationID"] == 123


     

def test_init_clean_df():
    """ test for init_clean_df function in data_processing.py
    """    
    import pandas as pd
    from jfk_taxis import data_processing
    

    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2011-05-25 14:28:00", 
            "2019-12-31 23:50:00",
            "2020-01-01 00:10:00",
            "2020-07-30 15:30:00"
        ]),
        "PULocationID": [1, 2, 3, 123],
    })

    cleaned_df = data_processing.init_clean_df(df, 2020)

    assert all(cleaned_df["tpep_pickup_datetime"].dt.year == 2020) # all returns true if all elts in iterable are true
    assert cleaned_df.shape[0] == 2


def test_select_jfk():
    """ test for select_jfk function in data_processing.py
    """    
    import pandas as pd
    from jfk_taxis import data_processing

    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2020-05-25 14:28:00", 
            "2020-12-31 23:50:00",
            "2020-01-01 00:10:00",
            "2020-07-30 15:30:00"
        ]),
        "PULocationID": [1, 132, 3, 132],
        "DOLocationID": [4, 5, 132, 7],
    })

    jfk_df = data_processing.select_jfk(df)

    assert jfk_df.shape[0] == 2
    assert all(jfk_df["PULocationID"].isin([132]))

def test_to_utc_hourly():
    """ test for to_utc_hourly function in data_processing.py

    NYC is UTC-5 during late autumn/winter and UTC-4 during spring/summer/early autumn https://en.wikipedia.org/wiki/Eastern_Time_Zone
    """    
    import pandas as pd
    from pandas import Timestamp
    from jfk_taxis import data_processing

    series = pd.Series([
        "2011-05-25 14:28:00",
        "2019-12-31 21:50:00",
        "2024-08-23 08:10:00"
    ])  

    converted_series = data_processing.to_utc_hourly(series)

    assert converted_series.dtype == "datetime64[ns, UTC]"
    assert converted_series.iloc[0] == Timestamp("2011-05-25 18:28:00+0000", tz='UTC')  # UTC-4
    assert converted_series.iloc[1] == Timestamp("2020-01-01 02:50:00+0000", tz='UTC')  # UTC-5
    assert converted_series.iloc[2] == Timestamp("2024-08-23 12:10:00+0000", tz='UTC')  # UTC-4

def test_convert_to_NYC():
    """ test for convert_to_NYC function in data_processing.py

    Combined with the previous test this shows that our functions are inverses of each other. 
    """    
    import pandas as pd
    from pandas import Timestamp
    from jfk_taxis import data_processing

    series = pd.Series(
        data = [200, 147, 23],
        index = ["2011-05-25 18:28:00+00:00", "2020-01-01 02:50:00+00:00", "2024-08-23 12:10:00+00:00"]  
    )

    converted_series = data_processing.convert_to_NYC(series)

    assert converted_series.index.dtype == "datetime64[ns, America/New_York]"
    assert converted_series.index[0] == Timestamp("2011-05-25 14:28:00-0400", tz='America/New_York')  # UTC-4
    assert converted_series.index[1] == Timestamp("2019-12-31 21:50:00-0500", tz='America/New_York')  # UTC-5
    assert converted_series.index[2] == Timestamp("2024-08-23 08:10:00-0400", tz='America/New_York')  # UTC-4

def test_create_ts():
    """ test for create_ts function in data_processing.py
    """    
    import pandas as pd
    from jfk_taxis import data_processing

    # Note this is in NYC time and will be converted to UTC for the time series
    df = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "2020-05-25 14:28:00", 
            "2020-05-25 23:50:00",
            "2020-05-26 00:10:00",
            "2020-05-26 15:30:00",
            "2020-05-28 14:45:00",
            "2020-05-28 15:05:00",
            "2020-05-28 15:15:00",
        ]),
        "PULocationID": [1, 132, 3, 132, 132, 132, 4],
        "DOLocationID": [4, 5, 132, 7, 8, 9, 10],
    })
    # Daily ts
    ts_daily = data_processing.create_ts(df, "daily")

    # Hourly ts
    ts_hourly = data_processing.create_ts(df, "hour")

    # Basic checks
    assert isinstance(ts_daily.index, pd.DatetimeIndex)
    assert isinstance(ts_hourly.index, pd.DatetimeIndex)
    assert ts_daily.index.freq == "D"
    assert ts_hourly.index.freq == "h"

    # Check data processing
    assert ts_daily.shape[0] == 4 # four days in data 25th to 28th May
    assert ts_daily.sum() == df.shape[0] # total trips should be same as number of rows in df
    assert ts_daily.iloc[0] == 1 # one trip on 25th May in UTC (as 14:28 in NYC is 18:28 in UTC but 23:50 is 04:50 on 26th May in UTC)
    assert ts_daily.iloc[1] == 3 # two trips on 26th May
    assert ts_daily.iloc[2] == 0 # no trips on 27th May
    assert ts_daily.iloc[3] == 3 # three trips on 28th May

    assert ts_hourly.shape[0] == 74 # 74 hours in data, 11 in 25th, 24 in 26th 27th and 15 in 28th
    assert ts_hourly.sum() == df.shape[0] # total trips should be same as number of rows in df
    print(ts_hourly)
    print(type(ts_hourly.index))
    assert ts_hourly["2020-05-25 18:00:00+00:00"] == 1 # 14:00 in NYC (UTC-4 in May) is 18:00 in UTC so this trip counts for 25th May 18:00 hour
    assert ts_hourly["2020-05-26 04:00:00+00:00"] == 1 # 23:50 in NYC is 04:50 in UTC so this trip counts for 26th May 04:00 hour
    assert ts_hourly["2020-05-26 19:00:00+00:00"] == 1 # 15:30 in NYC is 19:30 in UTC so this trip counts for 26th May 19:00 hour
    assert ts_hourly["2020-05-28 18:00:00+00:00"] == 1 # 14:45 in NYC is 18:45 in UTC so this trip counts for 28th May 18:00 hour
    assert ts_hourly["2020-05-28 19:00:00+00:00"] == 2 # 15:05 and 15:15 in NYC is 19:05 and 19:15 in UTC so these trips counts for 28th May 19:00 hour 

def test_combine_ts():
    """ test for combine_ts function in data_processing.py
    """    
    import pandas as pd
    import os
    import shutil
    from jfk_taxis import load_config
    from jfk_taxis import data_processing
    # Get config
    config, PROJECT_ROOT = load_config()

    # Constants for dir and file names
    # Data directories
    DATA_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"] 
    DATA_SAVE = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["processed_path"]

    # Saving conventions
    ORIGINAL_PARQUET_PREFIX = config["saving"]["original_parquet_prefix"]
    JFK_PARQUET_PREFIX = config["saving"]["jfk_parquet_prefix"]
    TS_PREFIX = config["saving"]["ts_prefix"] 
    TS_DAILY = config["saving"]["ts_daily"] 
    TS_HOURLY = config["saving"]["ts_hourly"]

    df_daily_2020 = pd.DataFrame({
        "pickup_date": pd.to_datetime([
            "2020-05-25 00:00:00+00:00",
            "2020-08-31 00:00:00+00:00",
            "2021-01-01 00:00:00+00:00", # this is due to our time conversions (as we work in UTC but the time is NYC time) so the first day of the next year appears in each of the daily series
        ]),
        "trips": [5, 10, 2]
    })
    df_daily_2021 = pd.DataFrame({
        "pickup_date": pd.to_datetime([
            "2021-01-01 00:00:00+00:00",
            "2021-02-17 00:00:00+00:00",
            "2021-10-28 00:00:00+00:00",
        ]),
        "trips": [2, 7, 1]
    })

    df_hourly_2020 = pd.DataFrame({
        "dt": pd.to_datetime([
            "2020-05-25 14:00:00+00:00",
            "2020-08-31 23:00:00+00:00",
            "2021-01-01 03:00:00+00:00", # same reason as before we get time stamps up to 4am on 1st Jan 2021 (as NYC is UTC-5 on 31st Dec)
        ]),
        "trips": [1, 3, 1]
    })

    df_hourly_2021 = pd.DataFrame({
        "dt": pd.to_datetime([
            "2021-01-01 03:00:00+00:00", # even tho in the actual time series we start from 5am which avoids overalp, we have added this for testing the groupby and then sum
            "2021-02-17 12:00:00+00:00",
            "2021-10-28 18:00:00+00:00", 
        ]),
        "trips": [1, 2, 1]
    })

    # To avoid overwriting the actual time series we will very quickly move the the real time series to a temporary folder and then move them back when done (note daily and hourly for 2020-2021 doesn't exist as in the notebooks we do 2011-2025 so don't have to move it)
    os.makedirs(DATA_SAVE / "temp", exist_ok=True) 
    shutil.move(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2020.csv", DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_DAILY}2020.csv")
    shutil.move(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2020.csv", DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_HOURLY}2020.csv")
    shutil.move(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2021.csv", DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_DAILY}2021.csv")
    shutil.move(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2021.csv", DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_HOURLY}2021.csv")

    # Save the created time series
    df_daily_2020.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2020.csv", index=False)
    df_daily_2021.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2021.csv", index=False)
    df_hourly_2020.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2020.csv", index=False)
    df_hourly_2021.to_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2021.csv", index=False)

    years = [2020, 2021]

    # Run the combine_ts function
    data_processing.combine_ts(years)

    # Reload these time series and check they are as expected
    df_daily = pd.read_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}{years[0]}-{years[-1]}.csv")
    df_daily["pickup_date"] = pd.to_datetime(df_daily["pickup_date"])

    df_hourly = pd.read_csv(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{years[0]}-{years[-1]}.csv")
    df_hourly["dt"] = pd.to_datetime(df_hourly["dt"])

    # Basic checks
    assert df_daily.shape[0] == 5 # 25th May, 31st Aug, 1st Jan, 17th Feb, 28th Oct
    assert df_daily["trips"].sum() == 27 # total trips should be same as total number of trips from both df
    assert df_daily["trips"].iloc[0] == 5
    assert df_daily["trips"].iloc[1] == 10
    assert df_daily["trips"].iloc[2] == 4 # 2 from each year combined
    assert df_daily["trips"].iloc[3] == 7
    assert df_daily["trips"].iloc[4] == 1

    assert df_hourly.shape[0] == 5 # 25th May 14:00, 31st Aug 23:00, 1st Jan 03:00, 17th Feb 12:00, 28th Oct 18:00
    assert df_hourly["trips"].sum() == 9 # total trips should be same as total number of trips from both df
    assert df_hourly["trips"].iloc[0] == 1
    assert df_hourly["trips"].iloc[1] == 3
    assert df_hourly["trips"].iloc[2] == 2 # 1 from each year combined
    assert df_hourly["trips"].iloc[3] == 2
    assert df_hourly["trips"].iloc[4] == 1

    # Clean up the test files we created
    import os
    os.remove(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2020.csv")
    os.remove(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2021.csv")
    os.remove(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2020.csv")
    os.remove(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2021.csv")
    os.remove(DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}{years[0]}-{years[-1]}.csv")
    os.remove(DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{years[0]}-{years[-1]}.csv")

    # Move back the original time series
    shutil.move(DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_DAILY}2020.csv", DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2020.csv")
    shutil.move(DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_HOURLY}2020.csv", DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2020.csv")
    shutil.move(DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_DAILY}2021.csv", DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}2021.csv")
    shutil.move(DATA_SAVE / "temp" / f"{TS_PREFIX}_{TS_HOURLY}2021.csv", DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}2021.csv")
    os.rmdir(DATA_SAVE / "temp")


def test_process_taxi_data():
    """ test for process_taxi_data function in data_processing.py
    """
    import pandas as pd
    import os
    from jfk_taxis import load_config
    from jfk_taxis import data_processing

    # Get config
    config, PROJECT_ROOT = load_config()

    # Constants for dir and file names
    # Data directories
    DATA_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]
    DATA_SAVE = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["processed_path"]

    # Saving conventions
    ORIGINAL_PARQUET_PREFIX = config["saving"]["original_parquet_prefix"]
    JFK_PARQUET_PREFIX = config["saving"]["jfk_parquet_prefix"]
    TS_PREFIX = config["saving"]["ts_prefix"]
    TS_DAILY = config["saving"]["ts_daily"]
    TS_HOURLY = config["saving"]["ts_hourly"]

    YEAR = 1976

    # Create minimal raw parquet inputs for the year
    # Mix of JFK (132) and non-JFK pickups across two months
    df_1976_01 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1976-01-05 10:15:00",
            "1976-01-05 23:50:00",
            "1976-01-06 00:10:00",
            "1976-01-07 15:30:00",
        ]),
        "PULocationID": [132, 1, 132, 5],
        "DOLocationID": [4, 5, 6, 7]
    })

    df_1976_07 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1976-07-02 14:28:00",
            "1976-07-12 23:50:00",
            "1976-07-26 00:10:00",
            "1976-07-30 15:30:00",
        ]),
        "PULocationID": [100, 132, 46, 132],
        "DOLocationID": [3, 55, 56, 9]
    })

    # Save files with slightly differing naming formats (to match real-world variance)
    raw_f1 = DATA_DIR / "yellow_tripdata_1976-01.parquet"
    raw_f2 = DATA_DIR / "yellow_trip____HELLOOOOOOOO!_data_1976-07.parquet"
    df_1976_01.to_parquet(raw_f1, index=False)
    df_1976_07.to_parquet(raw_f2, index=False)

    try:
        # Run the processing pipeline
        data_processing.process_taxi_data([YEAR], ["hour", "daily"])

        # Expected output paths
        parquet_original_path = DATA_SAVE / f"{ORIGINAL_PARQUET_PREFIX}_{YEAR}.parquet"
        parquet_jfk_path = DATA_SAVE / f"{JFK_PARQUET_PREFIX}_{YEAR}.parquet"
        ts_daily_path = DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}{YEAR}.csv"
        ts_hourly_path = DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{YEAR}.csv"

        # Basic existence checks
        assert parquet_original_path.exists()
        assert parquet_jfk_path.exists()
        assert ts_daily_path.exists()
        assert ts_hourly_path.exists()

        # Load outputs and validate content
        df_jfk = pd.read_parquet(parquet_jfk_path)
        assert not df_jfk.empty
        assert (df_jfk["PULocationID"] == 132).all()

        # Count of JFK trips in inputs
        expected_jfk_trips = (df_1976_01["PULocationID"] == 132).sum() + (df_1976_07["PULocationID"] == 132).sum()

        df_daily = pd.read_csv(ts_daily_path)
        df_hourly = pd.read_csv(ts_hourly_path)

        # Ensure index columns are present and trip totals match expected JFK count
        assert set(["pickup_date", "trips"]).issubset(df_daily.columns)
        assert set(["dt", "trips"]).issubset(df_hourly.columns)

        assert df_daily["trips"].sum() == expected_jfk_trips
        assert df_hourly["trips"].sum() == expected_jfk_trips
    finally:
        # Clean up created raw and processed files
        if os.path.exists(raw_f1):
            os.remove(raw_f1)
        if os.path.exists(raw_f2):
            os.remove(raw_f2)

        # Outputs
        out_files = [
            DATA_SAVE / f"{ORIGINAL_PARQUET_PREFIX}_{YEAR}.parquet",
            DATA_SAVE / f"{JFK_PARQUET_PREFIX}_{YEAR}.parquet",
            DATA_SAVE / f"{TS_PREFIX}_{TS_DAILY}{YEAR}.csv",
            DATA_SAVE / f"{TS_PREFIX}_{TS_HOURLY}{YEAR}.csv",
        ]
        for p in out_files:
            if os.path.exists(p):
                os.remove(p)


def test_taxi_data_visuals():
    """ test for taxi_data_visuals function in data_processing.py
    """
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from jfk_taxis import load_config
    from jfk_taxis import data_processing

    # Get config and paths
    config, PROJECT_ROOT = load_config()
    DATA_DIR = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]

    YEAR = 1977

    # Create minimal raw parquet input
    df_1977_01 = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime([
            "1977-01-05 10:15:00",
            "1977-01-05 23:50:00",
            "1977-01-06 00:10:00",
        ]),
        "PULocationID": [132, 1, 132],
        "DOLocationID": [4, 5, 6]
    })

    raw_f = DATA_DIR / "yellow_tripdata_1977-01.parquet"
    df_1977_01.to_parquet(raw_f, index=False)

    # Stub plt.show to avoid GUI blocking
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Should run without raising
        data_processing.taxi_data_visuals([YEAR])
    finally:
        # Restore show and cleanup
        plt.show = old_show
        if os.path.exists(raw_f):
            os.remove(raw_f)


def test_ts_plots():
    """ test for ts_plots function in data_processing.py
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from jfk_taxis import data_processing

    # Use non-interactive backend and stub show, essentially so that trying to show plots doesn't interupt tests
    # we just want to test that the functions run the plots have been checked manually
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Daily plot input: timezone-naive strings (as would come from CSV)
        df_daily = pd.DataFrame({
            "pickup_date": [
                "2020-05-25 00:00:00+00:00",
                "2020-05-26 00:00:00+00:00",
                "2020-05-28 00:00:00+00:00",
            ],
            "trips": [1, 3, 2]
        })

        # Should run without error
        data_processing.ts_plots(df_daily, feature="daily", year=2020, month=[])

        # Cloase all plots to avoid warnings
        plt.close("all")

        # Hourly plot input: timezone-naive strings (as would come from CSV)
        df_hour = pd.DataFrame({
            "dt": [
                "2020-05-25 18:00:00+00:00",
                "2020-05-26 04:00:00+00:00",
                "2020-05-28 19:00:00+00:00",
            ],
            "trips": [1, 1, 2]
        })

        data_processing.ts_plots(df_hour, feature="hour", year=2020, month=[5, 6])
    finally:
        plt.show = old_show


def test_plot_full_ts():
    """ test for plot_full_ts function in data_processing.py
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from jfk_taxis import data_processing

    # Non-interactive backend and stub show
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        df_daily = pd.DataFrame({
            "pickup_date": [
                "2020-05-25 00:00:00+00:00",
                "2020-05-26 00:00:00+00:00",
                "2021-01-01 00:00:00+00:00",
            ],
            "trips": [1, 3, 2]
        })

        data_processing.plot_full_ts(df_daily, years=[2020, 2021])
    finally:
        plt.show = old_show