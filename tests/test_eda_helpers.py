"""
test_eda_helpers.py
=========================

Unit tests for eda_helpers.py, again we won't be testing the plotting functions (I have checked them manually anyway), we also won't test load_geo_data_and_zone_lookup_app as this just loads files and converts
to EPSG 4326 for folium. We would know if this didn't work from the notebooks. 
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

