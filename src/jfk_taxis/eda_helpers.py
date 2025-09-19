"""
eda_helpers
=================

This module contains helper functions for exploratory data analysis and visualization of the taxi data. Most functions revolve around make_choropleth
which creates a choropleth map of the taxi zones with the number of trips in each zone.
"""


# --- Imports ---
import folium
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd
from .loading_helpers import load_config

# --- Load config ---
config, PROJECT_ROOT = load_config()

# --- Constants and Paths ---
DATA_DIR_MAPS = PROJECT_ROOT / config["data"]["reports_path"] / config["data"]["maps_path"]
DATA_DIR_RAW = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]
DATA_DIR_APP = PROJECT_ROOT / config["data"]["app_path"] / config["data"]["app_data_path"]

# --- Functions ---
def make_choropleth(df: pd.DataFrame, count_col: str, geo_data: gpd.GeoDataFrame, zone_lookup: pd.DataFrame, extra: str, scale: list, drop_rows: bool) -> folium.Map:
    """ Creates a choropleth map of the taxi zones with the number of trips in each zone

    Args:
        df (pd.DataFrame): dataframe containing the taxi trips
        count_col (str): is the column to count, either PULocationID or DOLocationID
        geo_data (gpd.GeoDataFrame): geopandas dataframe containing the taxi zones shapefile
        zone_lookup (pd.DataFrame): dataframe containing the taxi zone lookup file (how to match location id to zone name an borough)
        extra (str): extra string to add to the legend name
        scale (list): scale to use for the choropleth, if None folium will create its own scale
        drop_rows (bool): whether to drop rows in geo_data with trips over the scale max

    Returns:
        folium.Map: folium map object
    """    

    # Count trips in each zone
    trips_count = df[count_col].value_counts().reset_index()
    trips_count.columns = ["LocationID", "trips"]
    trips_count = trips_count.sort_values("trips", ascending=False)
    #display(trips_count.head())

    if drop_rows:
        smax = scale[-1]
        # Exclude rows over the scale max
        trips_count = trips_count[trips_count["trips"] <= smax]
    
    # Merge this into the orignal GeoPandasDataFrame so that we can use the trips count in the tooltip
    zones =  geo_data.merge(
        trips_count,
        left_on = "LocationID",
        right_on = "LocationID",
        how = "left"
    )

    # Either fill missing trips with 0 or drop rows with missing trips
    if drop_rows:
        # Drop missing trip counts   
        zones["trips"] = zones["trips"].dropna()
    else:
        # Fill missing trips with 0 
        zones["trips"] = zones["trips"].fillna(0)

    # We would also like to see the taxi service zone on the tooltip: Yellow, Green etc
    # To do this we will merge the taxi_zone_lookup onto zones as well
    service_zones = zone_lookup[["LocationID", "service_zone"]]

    zones = zones.merge(
        service_zones,
        left_on = "LocationID", 
        right_on = "LocationID",
        how = "left"
    )

    # Fill missing zones with "NA"
    zones["service_zone"] = zones["service_zone"].fillna("NA")

    # Create a base map
    m = folium.Map(location = [40.7128, -74.0060], zoom_start = 11)

    # Either we use the scale passed or let folium create its own
    if scale is None:
        # Add a choropleth map to base map
        ch = folium.Choropleth(
            geo_data = zones,
            data = trips_count,
            columns = ["LocationID", "trips"],
            key_on = "feature.properties.LocationID",
            fill_color = "YlGnBu", 
            legend_name = count_col + " counts by zone" + extra
        ).add_to(m)
    else:
        # Add a choropleth map to base map
        ch = folium.Choropleth(
            geo_data = zones,
            data = trips_count,
            columns = ["LocationID", "trips"],
            key_on = "feature.properties.LocationID",
            fill_color = "YlGnBu", 
            bins = scale,
            legend_name = count_col + " counts by zone" + extra
        ).add_to(m)
       # print("Used custom scale:", scale)

    # Add hover tooltip to the choropleth's polygons
    folium.features.GeoJsonTooltip(
        fields = ["zone", "borough", "trips", "service_zone", "LocationID"],
        aliases = ["Zone:", "Borough:", "Trips:", "Service Zone:", "Location ID:"],
        sticky = False
    ).add_to(ch.geojson)

    # Return map
    return m

def make_borough_mask_df(zone_lookup: pd.DataFrame, df: pd.DataFrame, drop: str, trip_type: str) -> pd.DataFrame:
    """ Drops rows in df where either PULocationID or DOLocationID is in the borough given by drop

    Args:
        zone_lookup (pd.DataFrame): dataframe containing the taxi zone lookup file (how to match location id to zone name and borough)
        df (pd.DataFrame): dataframe containing the taxi trips
        drop (str): borough to drop, e.g. "Manhattan"
        trip_type (str): is the column to check, either PULocationID or DOLocationID

    Returns:
        pd.DataFrame: dataframe with rows dropped
    """     

    # We need to find the location ids of the boroughs from drop
    id_list = zone_lookup.loc[zone_lookup["Borough"] == drop, "LocationID"].unique() 

    # Create an empty mask
    combined_mask = pd.Series([False] * len(df), index=df.index)
    
    # Create a mask for all rows with these locations ids
    for id in id_list:
        id = int(id) 

        mask = df[f"{trip_type}LocationID"] == id
        combined_mask = combined_mask | mask 
     
    # Keeps rows where the mask is false
    df = df[~combined_mask]

    return df

def make_borough_mask_geo_data(geo_data: gpd.GeoDataFrame, drop: str) -> gpd.GeoDataFrame:
    """ Drops rows in geo_data where the borough is given by drop

    Args:
        geo_data (gpd.GeoDataFrame): geoDataFrame containing the geographical data
        drop (str): borough to drop, e.g. "Manhattan"

    Returns:
        gpd.GeoDataFrame: geoDataFrame with rows dropped
    """    

    # Drop the rows in geo_data from drop
    mask = geo_data["borough"] == drop

    # Keeps geo_data where the mask is false
    geo_data = geo_data[~mask]
   
    return geo_data

def drop_id_geo_data(geo_data: gpd.GeoDataFrame, drop: list) -> gpd.GeoDataFrame:
    """ Drops rows in geo_data where the location id is in drop

    Args:
        geo_data (gpd.GeoDataFrame): geoDataFrame containing the geographical data
        drop (list): list of location ids to drop

    Returns:
        gpd.GeoDataFrame: geoDataFrame with rows dropped
    """    

    # Create an empty mask
    combined_mask = pd.Series([False] * len(geo_data), index=geo_data.index)
    
    # Create a mask for all rows with these locations ids
    for id in drop:
        id = int(id) 

        mask = geo_data["LocationID"] == id
        combined_mask = combined_mask | mask 
     
    # Keeps rows where the mask is false
    geo_data = geo_data[~combined_mask]

    return geo_data

def drop_id_df(df: pd.DataFrame, drop: list, trip_type: str) -> pd.DataFrame:
    """Drops rows in df where either PULocationID or DOLocationID has a location id in drop

    Args:
        df (pd.DataFrame): dataframe containing the taxi trips
        drop (list): list of location ids to drop
        trip_type (str): is the column to check, either PULocationID or DOLocationID

    Returns:
        pd.DataFrame: dataframe with rows dropped
    """    


    # Create an empty mask
    combined_mask = pd.Series([False] * len(df), index=df.index)
    
    # Create a mask for all rows with these locations ids
    for id in drop:
        id = int(id) 

        mask = df[f"{trip_type}LocationID"] == id
        combined_mask = combined_mask | mask 
     
    # Keeps rows where the mask is false
    df = df[~combined_mask]

    return df

def create_save_listed_adjusted_choropleths(geo_data: gpd.GeoDataFrame, zone_lookup: pd.DataFrame, extra: str, scale: list, years: list[int], months: list[int], drop_boroughs: list[str], drop_ids: list[int], save_file_suffix: str, drop_over_scale: bool) -> None:
    """ Create choropleth maps for each year and month in years and months, these we will be adjusted maps according to drop_boroughs and drop_ids.
    The maps will be saved as html files in the maps directory.

    Args:
        geo_data (gpd.GeoDataFrame): geopandas dataframe containing the taxi zones shape data
        zone_lookup (pd.DataFrame): dataframe containing the taxi zone lookup file (how to match location id to zone name and borough)
        extra (str): extra string to add to the legend name (e.g. no Manhattan)
        scale (list): scale to use for the choropleth, if None folium will create its own scale
        years (list[int]): list of years to create maps for
        months (list[int]): list of months to create maps for
        drop_boroughs (list[str]): list of boroughs to drop from the data
        drop_ids (list[int]): list of location ids to drop from the data
        save_file_suffix (str): suffix to add to the saved file names
        drop_over_scale (bool): whether to drop rows in geo_data with trips over the scale max
    """    

    geo_data = geo_data.copy() # To avoid modifying the original data

    # First drop the boroughs from the geo_data 
    for borough in drop_boroughs:
        geo_data = make_borough_mask_geo_data(geo_data, borough)

    # Then drop the location ids from the geo_data
    geo_data = drop_id_geo_data(geo_data, drop_ids)

    for year in years:
        for month in months:
            # Skip anything beyond config["eda"]["max_month_2025"] for 2025 (as doesn't exist)
            if (year == 2025) and (int(month) > config["eda"]["max_month_2025"]):
                continue

            # Load data frame for this year and month
            df = pd.read_parquet(DATA_DIR_RAW / f"yellow_tripdata_{year}-{month:02}.parquet")

            # We need two data frames, one where we drop in the pick ups, one where we drop in the drop offs
            df_pu = df.copy()
            df_do = df.copy()

            # Drop boroughs from the df
            for borough in drop_boroughs: 
                df_pu = make_borough_mask_df(zone_lookup, df_pu, borough, "PU")
                df_do = make_borough_mask_df(zone_lookup, df_do, borough, "DO")

            # Drop the ids from drop_zones in the df
            df_pu = drop_id_df(df_pu, drop_ids, "PU")
            df_do = drop_id_df(df_do, drop_ids, "DO")

            # Create pick up Choropleth
            m_3 = make_choropleth(df_pu, "PULocationID", geo_data, zone_lookup, f" {extra} {month} {year}", scale, drop_over_scale)

            # Export to HTML file
            m_3.save(DATA_DIR_MAPS / f"PULocationID_count_by_zone_{str(year)}_{month}_{save_file_suffix}.html")


            # Create drop off Choropleth
            m_4 = make_choropleth(df_do, "DOLocationID", geo_data, zone_lookup, f" {extra} {month} {year}", scale, drop_over_scale)

            # Export to HTML file
            m_4.save(DATA_DIR_MAPS / f"DOLocationID_count_by_zone_{str(year)}_{month}_{save_file_suffix}.html")

def multiplot_choropleths(geo_data: gpd.GeoDataFrame, scale: list[int], years: list[int], months: list[int]) -> None:
    """ Function creates choropleth maps for pick ups and drop offs for each year and month provided, in the same figure on a fixed scale.

    Args:
        geo_data (gpd.GeoDataFrame): geopandas dataframe containing the taxi zones shape data
        scale (list[int]): list of two integers defining the color scale for the choropleth maps
        years (list[int]): list of years to create maps for
        months (list[int]): list of months to create maps for
    """    

    # Set min and max of the scale
    smin = scale[0]
    smax = scale[-1]

    # Create two dicts of data frames, one for pick up trip counts and one for drop offs
    pu_dict = {}
    do_dict = {}

    # Loop through years and months to load the data frames and add to the dict
    for year in years:
        for month in months:
            # Skip anything beyond config["eda"]["max_month_2025"] for 2025 (as doesn't exist)
            if (year == 2025) and (int(month) > config["eda"]["max_month_2025"]):
                continue
            
            # Load data frame for this year and month
            df = pd.read_parquet(DATA_DIR_RAW / f"yellow_tripdata_{year}-{month:02}.parquet")

            # Count trips by pick ups and drop offs 
            for count_col in ["PULocationID", "DOLocationID"]:
                # Count trips in each zone
                trips_count = df[count_col].value_counts().reset_index()
                trips_count.columns = ["LocationID", "trips"]
                trips_count = trips_count.sort_values("trips", ascending=False)

                # Exclude rows over the scale max
                trips_count = trips_count[trips_count["trips"] <= smax]


                # Add to dict
                if count_col == "PULocationID":
                    pu_dict[f"{year}_{month}"] = trips_count
                else:
                    do_dict[f"{year}_{month}"] = trips_count

    # Compute ideal number of rows given we want 3 plots per row
    if len(pu_dict) % 3 == 0:
        num_row = len(pu_dict) // 3
    else:
        num_row = len(pu_dict) // 3 + 1    
    
    # Create the pick up plots
    fig_pu, axes_pu = plt.subplots(nrows =  num_row, ncols = 3, figsize = (20, num_row * 7))
    
    i = 0 # To track which axis we are on
    for year_month, trips_count in pu_dict.items(): 
        
        # Get axis
        ax = axes_pu[i // 3, i % 3] 

        # Merge this into the orignal GeoPandasDataFrame so that we can use the trips count in the tooltip
        zones =  geo_data.merge(
            trips_count,
            left_on = "LocationID",
            right_on = "LocationID",
            how = "left"
        )


        # Drop missing trip counts (this is because for smaller scales we are expecting manhattan for example to have missing trip counts)  
        zones["trips"] = zones["trips"].dropna()

        zones.plot(column = "trips", ax = ax, vmin = smin, vmax = smax, legend = True, cmap = "YlGnBu", edgecolor = "black")
        ax.set_title(f"Pick Ups {year_month}")

        # Increment i
        i += 1

    plt.show()

    fig_do, axes_do = plt.subplots(nrows =  num_row, ncols = 3, figsize = (20, num_row * 7))

    i = 0
    for year_month, trips_count in do_dict.items():

        # Get axis
        ax = axes_do[i // 3, i % 3] 
        
        # Merge this into the orignal GeoPandasDataFrame so that we can use the trips count in the tooltip
        zones =  geo_data.merge(
            trips_count,
            left_on = "LocationID",
            right_on = "LocationID",
            how = "left"
        )


        # Drop missing trip counts (this is because for smaller scales we are expecting manhattan for example to have missing trip counts)  
        zones["trips"] = zones["trips"].dropna()

        zones.plot(column = "trips", ax = ax, vmin = smin, vmax = smax, legend = True, cmap = "YlGnBu", edgecolor = "black")
        ax.set_title(f"Drop Offs {year_month}")

        i += 1

    plt.show()

def create_app_choropleths(geo_data: gpd.GeoDataFrame, zone_lookup: pd.DataFrame, extra: str, scale: list, year: int, month: int, pickup_or_drop_off: str, drop_boroughs: list[str], drop_ids: list[int]) -> folium.Map:
    """ Create choropleth maps for the app using the same approach as create_save_listed_adjusted_choropleths, now year and month will be singular values. Loads from app data dir.

    Args:
        geo_data (gpd.GeoDataFrame): geopandas dataframe containing the taxi zones shape data
        zone_lookup (pd.DataFrame): dataframe containing the taxi zone lookup file (how to match location id to zone name and borough)
        extra (str): extra string to add to the legend name (e.g. no Manhattan)
        scale (list): scale to use for the choropleth, if None folium will create its own scale
        year (int): year to create maps for
        month (int): month to create maps for
        pickup_or_drop_off (str): either "PU" for pick up or "DO" for drop off to indicate which map to create
        drop_boroughs (list[str]): list of boroughs to drop from the data
        drop_ids (list[int]): list of location ids to drop from the data
       

    Returns:
        folium.Map: the choropleth map to be displayed in the app
    """    

    geo_data = geo_data.copy() # To avoid modifying the original data



    # First drop the boroughs from the geo_data 
    for borough in drop_boroughs:
        geo_data = make_borough_mask_geo_data(geo_data, borough)

    # Then drop the location ids from the geo_data
    geo_data = drop_id_geo_data(geo_data, drop_ids)

    # Load data frame for this year and month
    df = pd.read_parquet(DATA_DIR_RAW / f"yellow_tripdata_{year}-{month:02}.parquet")

    # Drop boroughs from the df
    for borough in drop_boroughs: 
        df = make_borough_mask_df(zone_lookup, df, borough, pickup_or_drop_off)
    
    # Drop the ids from drop_zones in the df
    df = drop_id_df(df, drop_ids, pickup_or_drop_off)

    # Create Choropleth
    if pickup_or_drop_off == "PU":
        M = make_choropleth(df, "PULocationID", geo_data, zone_lookup, f" {extra} {month} {year}", scale, True)
    elif pickup_or_drop_off == "DO":
        M = make_choropleth(df, "DOLocationID", geo_data, zone_lookup, f" {extra} {month} {year}", scale, True)
    else:
        raise ValueError("pickup_or_drop_off must be either 'PU' or 'DO'")

    return M

def load_geo_data_and_zone_lookup_app() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """ Loads the geo data and zone lookup data from the app data directory

    Returns:
        tuple[gpd.GeoDataFrame, pd.DataFrame]: tuple containing the geo data and zone lookup data
    """    

    # Load the taxi zones shapefile using geopandas
    geo_data = gpd.read_file(DATA_DIR_APP / "taxi_zones.shp")

    # Reproject to EPSG 4326 for folium
    geo_data = geo_data.to_crs(epsg = 4326)

    # Load the taxi zone lookup file using pandas
    zone_lookup = pd.read_csv(DATA_DIR_APP / "taxi_zone_lookup.csv")

    return geo_data, zone_lookup


def create_rolling_average(size: int, daily_counts: pd.Series) -> None:
    """Creates a rolling average of the daily counts.

    Args:
        size (int): the window size for the rolling average.
        daily_counts (pd.Series): the daily counts to calculate the rolling average for.
    """    

    # Create rolling average
    moving_average = daily_counts.rolling(
        window = size,
        center = True,
        min_periods = size // 2,
    ).mean()

    ax = daily_counts.plot(style = ".", color = "0.5")
    moving_average.plot(
        ax = ax, linewidth = 3, title = f"JFK daily taxi trips - {size}-day moving average", legend = False,
    );

    # Show only one x axis tick per month
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.xticks(rotation = 45, ha = "right") 

    plt.show()




def create_rolling_average_hourly(size: int, hourly_counts: pd.Series) -> None:
    """Creates a rolling average of the hourly counts.

    Args:
        size (int): the window size for the rolling average.
        hourly_counts (pd.Series): the hourly counts to calculate the rolling average for.
    """    

    # Create rolling average
    moving_average = hourly_counts.rolling(
        window = size,
        center = True,
        min_periods = size // 2,
    ).mean()

    ax = hourly_counts.plot(style = ".", color = "0.5")
    moving_average.plot(
        ax = ax, linewidth = 3, title = f"JFK hourly taxi trips - {size}-hour moving average", legend = False,
    );

    # Show only one x axis tick per month
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.xticks(rotation = 45, ha = "right") 

    plt.show()


