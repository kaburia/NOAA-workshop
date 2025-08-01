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

# Converting CSV DataFrame to xarray.Dataset
def df_to_xarray(df: pd.DataFrame, values='total_rainfall') -> xr.Dataset:
    """
    Convert a DataFrame with columns "date", "lat", "lon", "total_rainfall" into an
    xarray.Dataset where the variable "total_rainfall" has dimensions ("time", "lat", "lon").
    This function pivots the data for each date into a 2D grid.
    """
    # Ensure that the date column is datetime
    df["date"] = pd.to_datetime(df["date"])
    # Get the sorted unique dates
    times = sorted(df["date"].unique())
    data_arrays = []
    for t in times:
        # Select data for this date
        df_t = df[df["date"] == t]
        # Pivot into a grid: rows=lat, columns=lon, values=total_rainfall
        pivot = df_t.pivot(index="lat", columns="lon", values=values)
        # Sort the rows and columns to ensure proper ordering
        pivot = pivot.sort_index(ascending=True)
        pivot = pivot.sort_index(axis=1, ascending=True)
        # Create a DataArray from the pivot table
        da = xr.DataArray(
            pivot.values,
            dims=("lat", "lon"),
            coords={"lat": pivot.index.values, "lon": pivot.columns.values}
        )
        data_arrays.append(da)
    # Concatenate along a new "time" dimension
    da_all = xr.concat(data_arrays, dim="time")
    da_all = da_all.assign_coords(time=times)
    # Create a Dataset with the variable name "total_rainfall"
    ds = xr.Dataset({values: da_all})
    return ds

def get_region_geojson(region_name, api_key):
    """
    Get the GeoJSON for a given region using the Google Maps Geocoding API.

    The function requests geocoding information for the specified region.
    It then extracts the bounding box (using the 'bounds' if available or
    the 'viewport' as a fallback) and converts it into a GeoJSON polygon.

    Args:
        region_name (str): The name of the region (e.g., "Nairobi").
        api_key (str): Your Google Maps API key.

    Returns:
        dict: A GeoJSON Feature representing the region's bounding box as a polygon.
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": region_name,
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if data['status'] != "OK":
        raise Exception("Error from API: " + data.get('status', 'Unknown error'))

    # Use the first result from the API response.
    result = data['results'][0]
    geometry = result['geometry']

    # Use 'bounds' if available; otherwise, fall back to 'viewport'
    if 'bounds' in geometry:
        bounds = geometry['bounds']
    else:
        bounds = geometry['viewport']

    # Extract southwest and northeast coordinates.
    sw = bounds['southwest']
    ne = bounds['northeast']

    # Create a polygon (closed ring) from the bounding box.
    polygon_coordinates = [
        [sw['lng'], sw['lat']],
        [sw['lng'], ne['lat']],
        [ne['lng'], ne['lat']],
        [ne['lng'], sw['lat']],
        [sw['lng'], sw['lat']]
    ]

    geojson_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [polygon_coordinates]
        },
        "properties": {
            "name": region_name
        }
    }

    return geojson_feature

def df_to_xarray2(df: pd.DataFrame, variables: list = None) -> xr.Dataset:
    """
    Convert a DataFrame with columns "date", "lat", "lon", and one or more data columns
    (e.g., "max_temperature", "min_temperature", "total_rainfall") into an xarray.Dataset.
    
    For each variable in `variables`, the data is pivoted into a 2D grid (lat x lon) for each date,
    then concatenated along a new "time" dimension. The resulting Dataset will have each variable
    as a separate DataArray with dimensions ("time", "lat", "lon").
    
    Parameters:
      df: pandas DataFrame containing the data.
      variables: List of variable column names to include. If None, defaults to ['total_rainfall'].
      
    Returns:
      xr.Dataset with variables as keys.
    """
    if variables is None:
        variables = ['total_rainfall']

    # Ensure that the date column is in datetime format.
    df["date"] = pd.to_datetime(df["date"])
    
    # Get the sorted unique dates.
    times = sorted(df["date"].unique())
    
    # Prepare a dictionary to hold the data arrays for each variable.
    data_vars = {}
    
    for var in variables:
        data_arrays = []
        for t in times:
            # Filter data for the current date.
            df_t = df[df["date"] == t]
            
            # Pivot into a grid: rows=lat, columns=lon, values=var.
            pivot = df_t.pivot(index="lat", columns="lon", values=var)
            
            # Sort the rows and columns for proper ordering.
            pivot = pivot.sort_index(ascending=True)
            pivot = pivot.sort_index(axis=1, ascending=True)
            
            # Create a DataArray from the pivot table.
            da = xr.DataArray(
                pivot.values,
                dims=("lat", "lon"),
                coords={"lat": pivot.index.values, "lon": pivot.columns.values}
            )
            data_arrays.append(da)
        
        # Concatenate the list of DataArrays along a new "time" dimension.
        da_all = xr.concat(data_arrays, dim="time")
        da_all = da_all.assign_coords(time=times)
        data_vars[var] = da_all

    # Create and return a Dataset containing all variables.
    ds = xr.Dataset(data_vars)
    return ds

# Get the min lat, min lon, max lon and max lat to form the bbox
def bbox_from_polygon(region_name, maps_key=''):
  # get the polygon from the region
  geojson = get_region_geojson(region_name, maps_key)
  if geojson is None:
    raise ValueError("Invalid region name")
  polygon = geojson['geometry']['coordinates'][0]
  # get the min lat, min lon, max lon and max lat to form the bbox
  min_lat = min(point[1] for point in polygon)
  min_lon = min(point[0] for point in polygon)
  max_lon = max(point[0] for point in polygon)
  max_lat = max(point[1] for point in polygon)
  return [min_lon, min_lat, max_lon, max_lat]


# Write xarray to netcdf file
def write_xarray_to_netcdf(ds, filename):
    """
    Write an xarray.Dataset to a NetCDF file.
    
    Parameters:
      ds: xarray.Dataset to write.
      filename: Name of the output NetCDF file.
    """
    ds.to_netcdf(filename)
    print(f"Dataset written to {filename}")



