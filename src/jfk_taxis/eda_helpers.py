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

# --- Functions ---
def make_choropleth(df: pd.DataFrame, count_col: str, geo_data: gpd.GeoDataFrame, zone_lookup: pd.DataFrame, extra: str, scale: list) -> folium.Map:
    """ Creates a choropleth map of the taxi zones with the number of trips in each zone

    Args:
        df (pd.DataFrame): dataframe containing the taxi trips
        count_col (str): is the column to count, either PULocationID or DOLocationID
        geo_data (gpd.GeoDataFrame): geopandas dataframe containing the taxi zones shapefile
        zone_lookup (pd.DataFrame): dataframe containing the taxi zone lookup file (how to match location id to zone name an borough)
        extra (str): extra string to add to the legend name
        scale (list): scale to use for the choropleth, if None folium will create its own scale

    Returns:
        folium.Map: folium map object
    """    

    # Count trips in each zone
    trips_count = df[count_col].value_counts().reset_index()
    trips_count.columns = ["LocationID", "trips"]
    trips_count = trips_count.sort_values("trips", ascending=False)
    #display(trips_count.head())
    
    # Merge this into the orignal GeoPandasDataFrame so that we can use the trips count in the tooltip
    zones =  geo_data.merge(
        trips_count,
        left_on = "LocationID",
        right_on = "LocationID",
        how = "left"
    )

    # Fill missing trip counts with 0 
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