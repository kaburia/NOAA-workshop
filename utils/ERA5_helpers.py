import ee
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
import math
import datetime
import io
from tqdm import tqdm
from datetime import datetime, timedelta
from IPython.display import HTML, display
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from filter_stations import retreive_data, Filter
import base64
import json
import requests
import datetime
from helpers import get_region_geojson, df_to_xarray



def extract_era5_daily(start_date_str, end_date_str, bbox=None, polygon=None, era5_l=False):
    """
    Extract ERA5 reanalysis data (daily aggregated) from Google Earth Engine for a given bounding box or polygon and time range.
    The extraction is performed on a daily basis by aggregating hourly images (using the mean) for each day.
    For each day, the function retrieves the ERA5 HOURLY images, aggregates them, adds pixel coordinate bands (longitude
    and latitude), and uses sampleRectangle to extract a grid of pixel values. The results for each variable (band) are then
    organized into pandas DataFrames with the following columns:
      - date: The daily timestamp (ISO formatted)
      - latitude: The latitude coordinate of the pixel center
      - longitude: The longitude coordinate of the pixel center
      - value: The aggregated pixel value for that variable

    Args:
        start_date_str (str): Start datetime in ISO format, e.g., '2020-01-01T00:00:00'.
        end_date_str (str): End datetime in ISO format, e.g., '2020-01-02T00:00:00'.
        bbox (list or tuple, optional): Bounding box specified as [minLon, minLat, maxLon, maxLat]. Default is None.
        polygon (list, optional): Polygon specified as a list of coordinate pairs (e.g., [[lon, lat], ...]).
                                  If provided, the polygon geometry will be used instead of the bounding box.

    Returns:
        dict: A dictionary where keys are variable (band) names and values are pandas DataFrames containing
              the daily aggregated data.
    """
    # Convert input datetime strings to Python datetime objects.
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S')
    end_date   = datetime.datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M:%S')

    # Define the geometry: Use polygon if provided, otherwise use bbox.
    if polygon is not None:
        region = ee.Geometry.Polygon(polygon)
    elif bbox is not None:
        region = ee.Geometry.Rectangle(bbox)
    else:
        raise ValueError("Either bbox or polygon must be provided.")

    # Define a scale in meters corresponding approximately to 0.25° (at the equator, 1° ≈ 111320 m).
    scale_m = 27830

    # This dictionary will accumulate extracted records for each variable (band).
    results = {}

    # Loop over each day in the specified time range.
    current = start_date
    while current < end_date:
        next_day = current + datetime.timedelta(days=1)

        # Format the current time window in ISO format.
        t0_str = current.strftime('%Y-%m-%dT%H:%M:%S')
        t1_str = next_day.strftime('%Y-%m-%dT%H:%M:%S')

        print(f"Processing {t0_str} to {t1_str}")

        # If ER5 Land (0.1) or ERA5 (0.25)
        if era5_l:
            # Get the ERA5 Land hourly image collection for the current day.
            collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
                            .filterDate(ee.Date(t0_str), ee.Date(t1_str))
        else:
            # Get the ERA5 hourly image collection for the current day.
            collection = ee.ImageCollection('ECMWF/ERA5/HOURLY') \
                            .filterDate(ee.Date(t0_str), ee.Date(t1_str))

        # Aggregate the hourly images into a single daily image using the mean.
        image = collection.mean()

        # Add bands containing the pixel longitude and latitude.
        image = image.addBands(ee.Image.pixelLonLat())

        # Use sampleRectangle to extract a grid of pixel values over the region.
        region_data = image.sampleRectangle(region=region, defaultValue=0).getInfo()

        # The pixel values for each band are in the "properties" dictionary.
        props = region_data['properties']

        # Extract the coordinate arrays from the added pixelLonLat bands.
        lon_array = props['longitude']  # 2D array of longitudes
        lat_array = props['latitude']   # 2D array of latitudes

        # Determine the dimensions of the extracted grid.
        nrows = len(lon_array)
        ncols = len(lon_array[0]) if nrows > 0 else 0

        # Identify the names of the bands that hold ERA5 variables, excluding the coordinate bands.
        band_names = [key for key in props.keys() if key not in ['longitude', 'latitude']]

        # Initialize results lists for each band if not already present.
        for band in band_names:
            if band not in results:
                results[band] = []

        # Loop over each pixel in the grid.
        for i in range(nrows):
            for j in range(ncols):
                pixel_lon = lon_array[i][j]
                pixel_lat = lat_array[i][j]
                # For each ERA5 variable band, extract the pixel value and create a record.
                for band in band_names:
                    pixel_value = props[band][i][j]
                    record = {
                        'date': t0_str,  # daily timestamp as a string
                        'latitude': pixel_lat,
                        'longitude': pixel_lon,
                        'value': pixel_value
                    }
                    results[band].append(record)

        # Advance to the next day.
        current = next_day

    # Convert the accumulated results for each band into pandas DataFrames.
    dataframes = {band: pd.DataFrame(records) for band, records in results.items()}
    return dataframes

# Extract ERA5 data based on the region and the start and end date
def era5_data_extracts(region_name, start_date, end_date, maps_key, bbox=None, era5_l=True):
  # Get the geojson region if bbox is None
  if bbox is None:
    geojson = get_region_geojson(region_name, maps_key)
    polygon = geojson['geometry']['coordinates'][0]
  else:
    polygon = None

  # Extract the data
  variable_dataframes_polygon = extract_era5_daily(start_date, end_date, polygon=polygon, bbox=bbox, era5_l=era5_l)
  return variable_dataframes_polygon


import copy
def era5_var_handling(variable_dataframes, variable_name, xarray_ds=False):
    # Create a deep copy of the input dictionary to avoid modifying the original
    variable_dataframes_copy = copy.deepcopy(variable_dataframes)

    # Handle 2m_temperature by converting from Kelvin to degrees Celsius
    if variable_name == 'temperature_2m':
        variable_dataframes_copy[variable_name]['value'] -= 273.15
    # Handle total_precipitation by multiplying by 24000
    elif variable_name == 'total_precipitation':
        variable_dataframes_copy[variable_name]['value'] *= 24000
    # Compute the wind speed from u and v components
    elif variable_name == 'wind_speed':
        variable_dataframes_copy[variable_name]['value'] = np.sqrt(
            variable_dataframes_copy[variable_name]['u_component_of_wind_10m']**2 +
            variable_dataframes_copy[variable_name]['v_component_of_wind_10m']**2
        )

    # Format the latitude and longitude columns
    variable_dataframes_copy[variable_name].rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    # Rename the value column to the variable name
    variable_dataframes_copy[variable_name].rename(columns={'value': variable_name}, inplace=True)

    # Convert to xarray if specified
    if xarray_ds:
        var_ds = df_to_xarray(variable_dataframes_copy[variable_name], values=variable_name)
        return var_ds
    else:
        return variable_dataframes_copy[variable_name]

