"""
test_eda_helpers.py
=========================

Unit tests for eda_helpers.py. The tests for a lot of the plotting functions are essentially smoke tests we have manually checked that the 
choropleths produce the expected values for various zones as well. 
"""

def test_make_borough_mask_df():
    """ test for make_borough_mask_df
    """    

    import pandas as pd
    from jfk_taxis import eda_helpers

    # Create zone lookup
    zone_lookup = pd.DataFrame({
        "LocationID": [1, 2, 3, 4, 5],
        "Borough": ["A", "A", "B", "B", "C"]
    })

    # Create test df, we will want to test dropping from both PULocationID and DOLocationID
    df = pd.DataFrame({
        "PULocationID": [1, 2, 3, 4, 5, 1, 3, 5],
        "DOLocationID": [5, 4, 3, 2, 1, 5, 3, 1]
    })

    # Run function, filter for boroughs A and C in both DOLocation ID and PULocationID as well as borough "D" which isn't in the data
    df_do_A = eda_helpers.make_borough_mask_df(zone_lookup, df, "A", "DO") # should be PU: [1, 2, 3, 1, 3] and DO: [5, 4, 3, 5, 3]
    df_pu_A = eda_helpers.make_borough_mask_df(zone_lookup, df, "A", "PU") # should be PU: [1, 2, 3, 1, 3] and DO: [5, 4, 3, 5, 3]
    df_do_C = eda_helpers.make_borough_mask_df(zone_lookup, df, "C", "DO") # should be PU: [1, 2, 3, 1, 3, 5] and DO: [5, 4, 3, 5, 3, 1]
    df_pu_C = eda_helpers.make_borough_mask_df(zone_lookup, df, "C", "PU") # should be PU: [1, 2, 3, 1, 3, 5] and DO: [5, 4, 3, 5, 3, 1]
    df_do_D = eda_helpers.make_borough_mask_df(zone_lookup, df, "D", "DO") # should be all rows as D not in data
    df_pu_D = eda_helpers.make_borough_mask_df(zone_lookup, df, "D", "PU") # should be all rows as D not in data 

    # Basic checks
    assert isinstance(df_do_A, pd.DataFrame)
    assert isinstance(df_pu_A, pd.DataFrame)
    assert isinstance(df_do_C, pd.DataFrame)
    assert isinstance(df_pu_C, pd.DataFrame)
    assert isinstance(df_do_D, pd.DataFrame)
    assert isinstance(df_pu_D, pd.DataFrame)
    assert df_do_A.shape[0] == 5
    assert df_do_A.shape[1] == 2
    assert df_pu_A.shape[0] == 5
    assert df_pu_A.shape[1] == 2
    assert df_do_C.shape[0] == 6
    assert df_do_C.shape[1] == 2
    assert df_pu_C.shape[0] == 6
    assert df_pu_C.shape[1] == 2
    assert df_do_D.shape[0] == 8
    assert df_do_D.shape[1] == 2
    assert all(df_do_A["PULocationID"] == pd.Series([1, 2, 3, 1, 3], dtype="int64", name = "PULocationID", index = [0, 1, 2, 5, 6]))
    assert all(df_do_A["DOLocationID"] == pd.Series([5, 4, 3, 5, 3], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 5, 6]))
    assert all(df_pu_A["PULocationID"] == pd.Series([3, 4, 5, 3, 5], dtype="int64", name = "PULocationID", index = [2, 3, 4, 6, 7]))
    assert all(df_pu_A["DOLocationID"] == pd.Series([3, 2, 1, 3, 1], dtype="int64", name = "DOLocationID", index = [2, 3, 4, 6, 7]))
    assert all(df_do_C["PULocationID"] == pd.Series([2, 3, 4, 5, 3, 5], dtype="int64", name = "PULocationID", index = [1, 2, 3, 4, 6, 7]))
    assert all(df_do_C["DOLocationID"] == pd.Series([4, 3, 2, 1, 3, 1], dtype="int64", name = "DOLocationID", index = [1, 2, 3, 4, 6, 7]))
    assert all(df_pu_C["PULocationID"] == pd.Series([1, 2, 3, 4, 1, 3], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3, 5, 6]))
    assert all(df_pu_C["DOLocationID"] == pd.Series([5, 4, 3, 2, 5, 3], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3, 5, 6]))
    assert all(df_do_D["PULocationID"] == pd.Series([1, 2, 3, 4, 5, 1, 3, 5], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3, 4, 5, 6, 7]))
    assert all(df_do_D["DOLocationID"] == pd.Series([5, 4, 3, 2, 1, 5, 3, 1], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3, 4, 5, 6, 7]))



def test_make_borough_mask_geo_data():
    """ test for make_borough_mask_geo_data
    """    
    import geopandas as gpd
    from jfk_taxis import eda_helpers
    import pandas as pd

    # Create geo data 
    geo_data = gpd.GeoDataFrame({
        "borough": ["A", "B", "A", "C"],
        "geometry": [None, None, None, None]  # Geometry is not important for this test
    })

    # Run function, filter for borough A and C
    geo_A = eda_helpers.make_borough_mask_geo_data(geo_data, "A") # should be 2 rows
    geo_C = eda_helpers.make_borough_mask_geo_data(geo_data, "C") # should be 3 rows
    geo_D = eda_helpers.make_borough_mask_geo_data(geo_data, "D") # should be 4 rows (not in data)

    # Basic checks
    assert isinstance(geo_A, gpd.GeoDataFrame)
    assert isinstance(geo_C, gpd.GeoDataFrame)
    assert isinstance(geo_D, gpd.GeoDataFrame)
    assert geo_A.shape[0] == 2
    assert geo_A.shape[1] == 2
    assert geo_C.shape[0] == 3
    assert geo_C.shape[1] == 2
    assert geo_D.shape[0] == 4
    assert geo_D.shape[1] == 2
    assert all(geo_A["borough"] == pd.Series(["B", "C"], dtype="object", name = "borough", index = [1, 3]))
    assert all(geo_C["borough"] == pd.Series(["A", "B", "A"], dtype="object", name = "borough", index = [0, 1, 2]))
    assert all(geo_D["borough"] == pd.Series(["A", "B", "A", "C"], dtype="object", name = "borough", index = [0, 1, 2, 3]))

def test_drop_id_geo_data():
    """ test for drop_id_geo_data
    """    
    import geopandas as gpd
    from jfk_taxis import eda_helpers
    import pandas as pd

    # Create geo data 
    geo_data = gpd.GeoDataFrame({
        "borough": ["A", "B", "A", "C"],
        "geometry": [None, None, None, None],  # Geometry is not important for this test
        "LocationID": [1, 2, 1, 4]
    })

    # Run function
    geo_drop_1 = eda_helpers.drop_id_geo_data(geo_data, [1])
    geo_drop_12 = eda_helpers.drop_id_geo_data(geo_data, [1, 2])
    geo_drop_none = eda_helpers.drop_id_geo_data(geo_data, [])

    # Basic checks
    assert isinstance(geo_drop_1, gpd.GeoDataFrame)
    assert isinstance(geo_drop_12, gpd.GeoDataFrame)
    assert isinstance(geo_drop_none, gpd.GeoDataFrame)
    assert geo_drop_1.shape[0] == 2
    assert geo_drop_1.shape[1] == 3
    assert geo_drop_12.shape[0] == 1
    assert geo_drop_12.shape[1] == 3
    assert geo_drop_none.shape[0] == 4
    assert geo_drop_none.shape[1] == 3
    assert all(geo_drop_1["borough"] == pd.Series(["B", "C"], dtype="object", name = "borough", index = [1, 3]))
    assert all(geo_drop_12["borough"] == pd.Series(["C"], dtype="object", name = "borough", index = [3]))
    assert all(geo_drop_none["borough"] == pd.Series(["A", "B", "A", "C"], dtype="object", name = "borough", index = [0, 1, 2, 3]))

def test_drop_id_df():
    """ test for drop_id_df
    """
    import pandas as pd
    from jfk_taxis import eda_helpers

    # Create test df, we will want to test dropping from both PULocationID and DOLocationID
    df = pd.DataFrame({
        "PULocationID": [1, 2, 3, 4],
        "DOLocationID": [5, 4, 3, 2],
    })

    # Run function drop ids 1, 1 and 2, 7, 7 and 3 and 2, and none
    df_do_drop_1 = eda_helpers.drop_id_df(df, [1], "DO") # should be PU: [1, 2, 3, 4] and DO: [5, 4, 3, 2]
    df_pu_drop_1 = eda_helpers.drop_id_df(df, [1], "PU") # should be PU: [2, 3, 4] and DO: [4, 3, 2]
    df_do_drop_12 = eda_helpers.drop_id_df(df, [1, 2], "DO") # should be PU: [1, 2, 3] and DO: [5, 4, 3]
    df_pu_drop_12 = eda_helpers.drop_id_df(df, [1, 2], "PU") # should be PU: [3, 4] and DO: [3, 2]
    df_do_drop_7 = eda_helpers.drop_id_df(df, [7], "DO") # should be PU: [1, 2, 3, 4] and DO: [5, 4, 3, 2]
    df_pu_drop_7 = eda_helpers.drop_id_df(df, [7], "PU") # should be PU: [1, 2, 3, 4] and DO: [5, 4, 3, 2]
    df_do_drop_732 = eda_helpers.drop_id_df(df, [7, 3, 2], "DO") # should be PU: [1, 2] and DO: [5, 4]
    df_pu_drop_732 = eda_helpers.drop_id_df(df, [7, 3, 2], "PU") # should be PU: [1, 4] and DO: [5, 2]
    df_do_drop_none = eda_helpers.drop_id_df(df, [], "DO") # should be PU: [1, 2, 3, 4] and DO: [5, 4, 3, 2]
    df_pu_drop_none = eda_helpers.drop_id_df(df, [], "PU") # should be PU: [1, 2, 3, 4] and DO: [5, 4, 3, 2]

    # Basic checks
    assert isinstance(df_do_drop_1, pd.DataFrame)
    assert isinstance(df_pu_drop_1, pd.DataFrame)
    assert isinstance(df_do_drop_12, pd.DataFrame)
    assert isinstance(df_pu_drop_12, pd.DataFrame)
    assert isinstance(df_do_drop_7, pd.DataFrame)
    assert isinstance(df_pu_drop_7, pd.DataFrame)
    assert isinstance(df_do_drop_732, pd.DataFrame)
    assert isinstance(df_pu_drop_732, pd.DataFrame)
    assert isinstance(df_do_drop_none, pd.DataFrame)
    assert isinstance(df_pu_drop_none, pd.DataFrame)
    assert df_do_drop_1.shape[0] == 4
    assert df_do_drop_1.shape[1] == 2
    assert df_pu_drop_1.shape[0] == 3
    assert df_pu_drop_1.shape[1] == 2
    assert df_do_drop_12.shape[0] == 3
    assert df_do_drop_12.shape[1] == 2
    assert df_pu_drop_12.shape[0] == 2
    assert df_pu_drop_12.shape[1] == 2
    assert df_do_drop_7.shape[0] == 4
    assert df_do_drop_7.shape[1] == 2
    assert df_pu_drop_7.shape[0] == 4
    assert df_pu_drop_7.shape[1] == 2
    assert df_do_drop_732.shape[0] == 2
    assert df_do_drop_732.shape[1] == 2
    assert df_pu_drop_732.shape[0] == 2
    assert df_pu_drop_732.shape[1] == 2
    assert df_do_drop_none.shape[0] == 4
    assert df_do_drop_none.shape[1] == 2
    assert df_pu_drop_none.shape[0] == 4
    assert df_pu_drop_none.shape[1] == 2
    assert all(df_do_drop_1["PULocationID"] == pd.Series([1, 2, 3, 4], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3]))
    assert all(df_do_drop_1["DOLocationID"] == pd.Series([5, 4, 3, 2], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3]))
    assert all(df_pu_drop_1["PULocationID"] == pd.Series([2, 3, 4], dtype="int64", name = "PULocationID", index = [1, 2, 3]))
    assert all(df_pu_drop_1["DOLocationID"] == pd.Series([4, 3, 2], dtype="int64", name = "DOLocationID", index = [1, 2, 3]))
    assert all(df_do_drop_12["PULocationID"] == pd.Series([1, 2, 3], dtype="int64", name = "PULocationID", index = [0, 1, 2]))
    assert all(df_do_drop_12["DOLocationID"] == pd.Series([5, 4, 3], dtype="int64", name = "DOLocationID", index = [0, 1, 2]))
    assert all(df_pu_drop_12["PULocationID"] == pd.Series([3, 4], dtype="int64", name = "PULocationID", index = [2, 3]))
    assert all(df_pu_drop_12["DOLocationID"] == pd.Series([3, 2], dtype="int64", name = "DOLocationID", index = [2, 3]))
    assert all(df_do_drop_7["PULocationID"] == pd.Series([1, 2, 3, 4], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3]))
    assert all(df_do_drop_7["DOLocationID"] == pd.Series([5, 4, 3, 2], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3]))
    assert all(df_pu_drop_7["PULocationID"] == pd.Series([1, 2, 3, 4], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3]))
    assert all(df_pu_drop_7["DOLocationID"] == pd.Series([5, 4, 3, 2], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3]))
    assert all(df_do_drop_732["PULocationID"] == pd.Series([1, 2], dtype="int64", name = "PULocationID", index = [0, 1]))
    assert all(df_do_drop_732["DOLocationID"] == pd.Series([5, 4], dtype="int64", name = "DOLocationID", index = [0, 1]))
    assert all(df_pu_drop_732["PULocationID"] == pd.Series([1, 4], dtype="int64", name = "PULocationID", index = [0, 3]))
    assert all(df_pu_drop_732["DOLocationID"] == pd.Series([5, 2], dtype="int64", name = "DOLocationID", index = [0, 3]))
    assert all(df_do_drop_none["PULocationID"] == pd.Series([1, 2, 3, 4], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3]))
    assert all(df_do_drop_none["DOLocationID"] == pd.Series([5, 4, 3, 2], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3]))
    assert all(df_pu_drop_none["PULocationID"] == pd.Series([1, 2, 3, 4], dtype="int64", name = "PULocationID", index = [0, 1, 2, 3]))
    assert all(df_pu_drop_none["DOLocationID"] == pd.Series([5, 4, 3, 2], dtype="int64", name = "DOLocationID", index = [0, 1, 2, 3]))


def test_make_choropleth():
    """ test for make_choropleth function in eda_helpers.py
    """
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Polygon
    from jfk_taxis import eda_helpers

    # Create test data with more varied trip counts to create enough categories for ColorBrewer
    df = pd.DataFrame({
        "PULocationID": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7],  # More trips to create varied counts
        "DOLocationID": [4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5]
    })

    # Create zone lookup
    zone_lookup = pd.DataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6, 7],
        "service_zone": ["Yellow", "Yellow", "Green", "Yellow", "Green", "Yellow", "Green"]
    })

    # Create mock geometry data - simple squares for each zone
    geometries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Zone 1
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Zone 2
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # Zone 3
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # Zone 4
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # Zone 5
        Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),  # Zone 6
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),  # Zone 7
    ]

    # Create geo data with proper geometries
    geo_data = gpd.GeoDataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6, 7],
        "zone": ["Zone1", "Zone2", "Zone3", "Zone4", "Zone5", "Zone6", "Zone7"],
        "borough": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens", "Bronx"],
        "geometry": geometries
    }, crs="EPSG:4326")

    # Test with PULocationID, custom scale (need at least 3 colors), drop_rows=True
    map_result = eda_helpers.make_choropleth(df, "PULocationID", geo_data, zone_lookup, " test", [0, 2, 4, 6], True)

    # Basic checks
    assert map_result is not None
    assert hasattr(map_result, 'save')  # Should be a folium Map object

    # Test with DOLocationID, no scale, drop_rows=False 
    map_result2 = eda_helpers.make_choropleth(df, "DOLocationID", geo_data, zone_lookup, " test2", None, False)

    assert map_result2 is not None
    assert hasattr(map_result2, 'save')


def test_create_rolling_average():
    """ test for create_rolling_average function in eda_helpers.py
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from jfk_taxis import eda_helpers

    # Use non-interactive backend and stub show
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Create test daily counts series
        daily_counts = pd.Series(
            data=[10, 15, 12, 18, 14, 20, 16, 22, 18, 25],
            index=pd.date_range('2020-01-01', periods=10, freq='D')
        )

        # Should run without error
        eda_helpers.create_rolling_average(3, daily_counts)
        
        # Close plots to avoid warnings
        plt.close("all")
    finally:
        plt.show = old_show


def test_create_rolling_average_hourly():
    """ test for create_rolling_average_hourly function in eda_helpers.py
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from jfk_taxis import eda_helpers

    # Use non-interactive backend and stub show
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Create test hourly counts series (use 'h' instead of 'H' to avoid deprecation warning)
        hourly_counts = pd.Series(
            data=[5, 8, 6, 9, 7, 10, 8, 11, 9, 12],
            index=pd.date_range('2020-01-01', periods=10, freq='h')
        )

        # Should run without error
        eda_helpers.create_rolling_average_hourly(5, hourly_counts)
        
        # Close plots to avoid warnings
        plt.close("all")
    finally:
        plt.show = old_show


def test_create_app_choropleths():
    """ test for create_app_choropleths function in eda_helpers.py
    """
    import pandas as pd
    import geopandas as gpd
    import os
    from shapely.geometry import Polygon
    from jfk_taxis import load_config
    from jfk_taxis import eda_helpers

    # Get config
    config, PROJECT_ROOT = load_config()
    DATA_DIR_RAW = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]

    # Create test parquet file with more varied trip counts
    test_df = pd.DataFrame({
        "PULocationID": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7],
        "DOLocationID": [4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5]
    })
    
    test_file = DATA_DIR_RAW / "yellow_tripdata_1978-01.parquet"
    test_df.to_parquet(test_file, index=False)

    # Create test data
    zone_lookup = pd.DataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6],
        "Borough": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"],
        "service_zone": ["Yellow", "Yellow", "Green", "Yellow", "Green", "Yellow"]
    })

    # Create mock geometry data - simple squares for each zone
    geometries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Zone 1
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Zone 2
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # Zone 3
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # Zone 4
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # Zone 5
        Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),  # Zone 6
    ]

    # Create geo data with proper geometries
    geo_data = gpd.GeoDataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6],
        "zone": ["Zone1", "Zone2", "Zone3", "Zone4", "Zone5", "Zone6"],
        "borough": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"],
        "geometry": geometries
    }, crs="EPSG:4326")

    try:
        # Test PU choropleth (need at least 3 colors for scale)
        map_pu = eda_helpers.create_app_choropleths(
            geo_data, zone_lookup, "test", [0, 2, 4, 10], 1978, 1, "PU", ["Manhattan"], [1]
        )
        assert map_pu is not None
        assert hasattr(map_pu, 'save')

        # Test DO choropleth
        map_do = eda_helpers.create_app_choropleths(
            geo_data, zone_lookup, "test", [0, 2, 4, 10], 1978, 1, "DO", [], []
        )
        assert map_do is not None
        assert hasattr(map_do, 'save')

    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)


def test_create_save_listed_adjusted_choropleths():
    """ test for create_save_listed_adjusted_choropleths function in eda_helpers.py
    """
    import pandas as pd
    import geopandas as gpd
    import os
    from shapely.geometry import Polygon
    from jfk_taxis import load_config
    from jfk_taxis import eda_helpers

    # Get config
    config, PROJECT_ROOT = load_config()
    DATA_DIR_RAW = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]
    DATA_DIR_MAPS = PROJECT_ROOT / config["data"]["reports_path"] / config["data"]["maps_path"]

    # Create test parquet files for 1979
    test_df = pd.DataFrame({
        "PULocationID": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7],
        "DOLocationID": [4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5]
    })
    
    test_files = [
        DATA_DIR_RAW / "yellow_tripdata_1979-01.parquet",
        DATA_DIR_RAW / "yellow_tripdata_1979-02.parquet"
    ]
    
    for test_file in test_files:
        test_df.to_parquet(test_file, index=False)

    # Create test data
    zone_lookup = pd.DataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6],
        "Borough": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"],
        "service_zone": ["Yellow", "Yellow", "Green", "Yellow", "Green", "Yellow"]
    })

    # Create mock geometry data - simple squares for each zone
    geometries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Zone 1
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Zone 2
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # Zone 3
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # Zone 4
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # Zone 5
        Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),  # Zone 6
    ]

    # Create geo data with proper geometries
    geo_data = gpd.GeoDataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6],
        "zone": ["Zone1", "Zone2", "Zone3", "Zone4", "Zone5", "Zone6"],
        "borough": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"],
        "geometry": geometries
    }, crs="EPSG:4326")

    # Define expected files outside try block to avoid UnboundLocalError
    expected_files = [
        DATA_DIR_MAPS / "PULocationID_count_by_zone_1979_1_test_suffix.html",
        DATA_DIR_MAPS / "DOLocationID_count_by_zone_1979_1_test_suffix.html",
        DATA_DIR_MAPS / "PULocationID_count_by_zone_1979_2_test_suffix.html",
        DATA_DIR_MAPS / "DOLocationID_count_by_zone_1979_2_test_suffix.html"
    ]

    try:
        # Run the function (need at least 3 colors for scale)
        eda_helpers.create_save_listed_adjusted_choropleths(
            geo_data, zone_lookup, "test", [0, 2, 4, 10], [1979], [1, 2], 
            ["Manhattan"], [1], "test_suffix", True
        )

        # Check that HTML files were created
        for expected_file in expected_files:
            assert os.path.exists(expected_file), f"Expected file {expected_file} was not created"

    finally:
        # Clean up test files
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # Clean up created HTML files
        for expected_file in expected_files:
            if os.path.exists(expected_file):
                os.remove(expected_file)


def test_load_geo_data_and_zone_lookup_app():
    """ test for load_geo_data_and_zone_lookup_app function in eda_helpers.py
    """
    import pandas as pd
    import geopandas as gpd
    import os
    import shutil
    from shapely.geometry import Polygon
    from jfk_taxis import load_config
    from jfk_taxis import eda_helpers

    # Get config
    config, PROJECT_ROOT = load_config()
    DATA_DIR_APP = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]

    # Create temporary directory for backup
    temp_dir = DATA_DIR_APP / "temp_test_backup"
    os.makedirs(temp_dir, exist_ok=True)

    # Files to backup 
    real_csv_file = DATA_DIR_APP / "taxi_zone_lookup.csv"
    
    # Additional shapefile components that need to be backed up
    shapefile_components = [
        "taxi_zones.shp",
        "taxi_zones.shx",
        "taxi_zones.dbf",
        "taxi_zones.prj"
    ]

    try:
        # Move real files to temp directory
        for component in shapefile_components:
            src = DATA_DIR_APP / component
            if os.path.exists(src):
                shutil.move(str(src), str(temp_dir / component))
        
        if os.path.exists(real_csv_file):
            shutil.move(str(real_csv_file), str(temp_dir / "taxi_zone_lookup.csv"))

        # Create fake test shapefile with proper geometries
        test_geometries = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        ]

        test_geo_data = gpd.GeoDataFrame({
            "LocationID": [1, 2, 3],
            "zone": ["TestZone1", "TestZone2", "TestZone3"],
            "borough": ["TestBorough1", "TestBorough2", "TestBorough3"],
            "geometry": test_geometries
        }, crs="EPSG:2263")  # Use a different CRS to test conversion to EPSG:4326

        # Save test shapefile
        test_geo_data.to_file(str(DATA_DIR_APP / "taxi_zones.shp"))

        # Create fake test zone lookup CSV
        test_zone_lookup = pd.DataFrame({
            "LocationID": [1, 2, 3],
            "Borough": ["TestBorough1", "TestBorough2", "TestBorough3"],
            "Zone": ["TestZone1", "TestZone2", "TestZone3"],
            "service_zone": ["Yellow", "Green", "Yellow"]
        })
        test_zone_lookup.to_csv(str(real_csv_file), index=False)

        # Run the function
        geo_data, zone_lookup = eda_helpers.load_geo_data_and_zone_lookup_app()

        # Validate results
        assert isinstance(geo_data, gpd.GeoDataFrame), "geo_data should be a GeoDataFrame"
        assert isinstance(zone_lookup, pd.DataFrame), "zone_lookup should be a DataFrame"
        
        # Check that CRS was converted to EPSG:4326
        assert geo_data.crs == "EPSG:4326", "geo_data should be converted to EPSG:4326"
        
        # Check that data was loaded correctly
        assert geo_data.shape[0] == 3, "geo_data should have 3 rows"
        assert zone_lookup.shape[0] == 3, "zone_lookup should have 3 rows"
        
        # Check column names
        assert "LocationID" in geo_data.columns
        assert "geometry" in geo_data.columns
        assert "LocationID" in zone_lookup.columns
        assert "Borough" in zone_lookup.columns

    finally:
        # Clean up test files
        for component in shapefile_components:
            test_file = DATA_DIR_APP / component
            if os.path.exists(test_file):
                os.remove(test_file)
        
        if os.path.exists(real_csv_file):
            os.remove(real_csv_file)

        # Restore real files from temp directory
        for component in shapefile_components:
            src = temp_dir / component
            if os.path.exists(src):
                shutil.move(str(src), str(DATA_DIR_APP / component))
        
        csv_backup = temp_dir / "taxi_zone_lookup.csv"
        if os.path.exists(csv_backup):
            shutil.move(str(csv_backup), str(real_csv_file))

        # Remove temp directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_multiplot_choropleths():
    """ test for multiplot_choropleths function in eda_helpers.py
    """
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import os
    from shapely.geometry import Polygon
    from jfk_taxis import load_config
    from jfk_taxis import eda_helpers

    # Get config
    config, PROJECT_ROOT = load_config()
    DATA_DIR_RAW = PROJECT_ROOT / config["data"]["data_path"] / config["data"]["raw_path"]

    # Create test parquet files for 1980 and 1981 - use multiple years and months
    test_df = pd.DataFrame({
        "PULocationID": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 6, 7],
        "DOLocationID": [4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5]
    })
    
    test_files = [
        DATA_DIR_RAW / "yellow_tripdata_1980-01.parquet",
        DATA_DIR_RAW / "yellow_tripdata_1980-02.parquet",
        DATA_DIR_RAW / "yellow_tripdata_1981-01.parquet",
        DATA_DIR_RAW / "yellow_tripdata_1981-02.parquet"
    ]
    
    for test_file in test_files:
        test_df.to_parquet(test_file, index=False)

    # Create mock geometry data - simple squares for each zone
    geometries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Zone 1
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # Zone 2
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # Zone 3
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # Zone 4
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),  # Zone 5
        Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),  # Zone 6
    ]

    # Create test geo data with proper geometries
    geo_data = gpd.GeoDataFrame({
        "LocationID": [1, 2, 3, 4, 5, 6],
        "zone": ["Zone1", "Zone2", "Zone3", "Zone4", "Zone5", "Zone6"],
        "borough": ["Manhattan", "Brooklyn", "Queens", "Manhattan", "Brooklyn", "Queens"],
        "geometry": geometries
    }, crs="EPSG:4326")

    # Use non-interactive backend and stub show
    import matplotlib
    matplotlib.use("Agg", force=True)
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Should run without error (use multiple years and months to test properly)
        eda_helpers.multiplot_choropleths(geo_data, [0, 10], [1980, 1981], [1, 2])
        
        # Close plots to avoid warnings
        plt.close("all")
    finally:
        plt.show = old_show
        
        # Clean up test files
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)