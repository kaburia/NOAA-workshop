def subset_stations_in_bbox(noaa_stations, bbox):
    """
    Subsets NOAA stations within a given bounding box.

    Args:
        stations metadata (pd.DataFrame): DataFrame of NOAA/TAHMO stations with 'Latitude' and 'Longitude' columns.
        bbox (list or tuple): Bounding box coordinates in the format [min_lon, min_lat, max_lon, max_lat].

    Returns:
        pd.DataFrame: DataFrame containing only the NOAA stations within the specified bounding box.
    """

    min_lat, min_lon, max_lat, max_lon = bbox

    # Use boolean indexing to efficiently filter the dataframe.
    subset = noaa_stations[
        (noaa_stations["Longitude"] >= min_lon) &
        (noaa_stations["Longitude"] <= max_lon) &
        (noaa_stations["Latitude"] >= min_lat) &
        (noaa_stations["Latitude"] <= max_lat)
    ]
    return subset