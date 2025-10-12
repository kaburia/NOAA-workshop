import ee
import pandas as pd
import datetime
from utils.helpers import get_region_geojson
import datetime
import ee
import os
import rioxarray
import xarray as xr

def extract_chirps_daily(start_date_str, end_date_str, bbox=None, region_name=None, polygon=None, api_key=''):
    """
    Extract CHIRPS daily precipitation data from Google Earth Engine for a given bounding box and time range.
    The extraction is performed on a daily basis. For each day, the function:
      - Filters the CHIRPS daily image collection for that day.
      - Adds pixel coordinate bands (longitude and latitude).
      - Uses sampleRectangle to extract a grid of pixel values.
      - Organizes the results into a pandas DataFrame with the following columns:
          - date: The daily timestamp (ISO formatted)
          - latitude: The latitude coordinate of the pixel center
          - longitude: The longitude coordinate of the pixel center
          - precipitation: The pixel value representing daily precipitation.

    Args:
        start_date_str (str): Start datetime in ISO format, e.g., '2020-01-01T00:00:00'.
        end_date_str (str): End datetime in ISO format, e.g., '2020-01-02T00:00:00'.
        bbox (list or tuple): Bounding box specified as [minLon, minLat, maxLon, maxLat].

    Returns:
        pd.DataFrame: A pandas DataFrame containing the daily precipitation data.
    """
    # Convert input datetime strings to Python datetime objects.
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M:%S')
    end_date   = datetime.datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M:%S')

    # Handle the input from bbox to polygon to region name
    if bbox is not None:
        region = ee.Geometry.Rectangle(bbox)
    elif region_name is not None:
        # get the region geojson
        region_geo = get_region_geojson(region_name, api_key)
        polygon = region_geo['features'][0]['geometry']['coordinates'][0]
        region = ee.Geometry.Polygon(polygon)
    elif polygon is not None:
        region = ee.Geometry.Polygon(polygon)
    # Define a scale in meters corresponding approximately to the CHIRPS resolution (~0.05°).
    scale_m = 5000  # You might adjust this value depending on your needs.

    # This list will accumulate extracted records.
    results = []

    # Loop over each day in the specified time range.
    current = start_date
    while current < end_date:
        next_day = current + datetime.timedelta(days=1)

        # Format the current time window in ISO format.
        t0_str = current.strftime('%Y-%m-%dT%H:%M:%S')
        t1_str = next_day.strftime('%Y-%m-%dT%H:%M:%S')

        print(f"Processing {t0_str} to {t1_str}")

        # Filter the CHIRPS daily image collection for the current day.
        # The CHIRPS daily collection is available as 'UCSB-CHG/CHIRPS/DAILY'.
        collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                        .filterDate(ee.Date(t0_str), ee.Date(t1_str))
        image = collection.first()

        # If no image is found for this day, skip to the next day.
        if image is None:
            current = next_day
            continue

        # Add bands containing the pixel longitude and latitude.
        image = image.addBands(ee.Image.pixelLonLat())

        # Use sampleRectangle to extract a grid of pixel values over the region.
        # The 'precipitation' band is part of the CHIRPS dataset.
        region_data = image.sampleRectangle(region=region, defaultValue=0).getInfo()

        # The pixel values for each band are in the "properties" dictionary.
        props = region_data['properties']

        # Extract the coordinate arrays from the added pixelLonLat bands.
        lon_array = props['longitude']  # 2D array of longitudes
        lat_array = props['latitude']   # 2D array of latitudes

        # Determine the dimensions of the extracted grid.
        nrows = len(lon_array)
        ncols = len(lon_array[0]) if nrows > 0 else 0

        # Loop over each pixel in the grid.
        for i in range(nrows):
            for j in range(ncols):
                pixel_lon = lon_array[i][j]
                pixel_lat = lat_array[i][j]
                # Extract the precipitation value.
                precip_value = props['precipitation'][i][j]
                record = {
                    'date': t0_str,  # daily timestamp as a string
                    'lat': pixel_lat,
                    'lon': pixel_lon,
                    'total_rainfall': precip_value
                }
                results.append(record)

        # Advance to the next day.
        current = next_day

    # Convert the accumulated results into a pandas DataFrame.
    df = pd.DataFrame(results)
    return df



def download_chirps_pentad_geotiff(start_date, end_date, region, out_dir="chirps_pentad"):
    """
    Download CHIRPS Pentad images from GEE as GeoTIFFs for a given region and date range.
    """
    os.makedirs(out_dir, exist_ok=True)
    collection = ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD') \
        .filterDate(start_date, end_date) \
        .select('precipitation')
    # Get list of images and their dates
    images = collection.toList(collection.size())
    n = images.size().getInfo()
    filepaths = []
    for i in range(n):
        img = ee.Image(images.get(i))
        date_str = img.date().format('YYYYMMdd').getInfo()
        fname = f"{out_dir}/chirps_pentad_{date_str}.tif"
        url = img.clip(region).getDownloadURL({
            'scale': 5000,
            'region': region.getInfo(),
            'format': 'GEO_TIFF'
        })
        # Download the file
        import requests
        r = requests.get(url)
        with open(fname, 'wb') as f:
            f.write(r.content)
        filepaths.append(fname)
    return filepaths

def load_geotiffs_as_xarray(filepaths):
    """
    Load a list of GeoTIFFs as a single xarray.DataArray (stacked along 'time').
    """
    arrays = []
    times = []
    for fp in filepaths:
        arr = rioxarray.open_rasterio(fp)
        arrays.append(arr)
        # Extract date from filename
        date_str = fp.split('_')[-1].replace('.tif', '')
        times.append(date_str)
    # Stack along new time dimension
    data = xr.concat(arrays, dim='time')
    data = data.assign_coords(time=times)
    return data

import ee
import geemap
import xarray as xr
import rioxarray
import os
import glob
import pandas as pd

# ee.Initialize(project='leafy-computing-310902')  # <-- change project if needed

def get_chirps_pentad_gee(start_date, end_date, region=None, export_path="chirps_pentad.nc", daily_pentad=True):
    """
    Extract CHIRPS pentad rainfall data from Google Earth Engine as an xarray.Dataset.

    Parameters
    ----------
    start_date : str (YYYY-MM-DD)
        Start date for extraction.
    end_date : str (YYYY-MM-DD)
        End date for extraction.
    region : ee.Geometry, optional
        Region of interest (polygon). If None, loads global dataset.
    export_path : str
        Path to save the NetCDF file.
    daily_pentad : bool
        True extracts pentad, False extracts daily

    Returns
    -------
    xarray.Dataset
        Dataset with variable 'precipitation' and dimensions (time, y, x).
    """
    # Load CHIRPS pentad dataset (mm/5-days)
    if daily_pentad: 
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
    else:
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        

    if region:
        chirps = chirps.map(lambda img: img.clip(region))

    # Export ImageCollection → GeoTIFF stack
    export_dir = "chirps_temp"
    os.makedirs(export_dir, exist_ok=True)

    geemap.ee_export_image_collection(
        chirps,
        out_dir=export_dir,
        scale=5500,  # ~5.5 km resolution
        file_per_band=False,
    )

    # Collect exported GeoTIFFs
    tiff_files = glob.glob(os.path.join(export_dir, "*.tif"))
    tiff_files.sort()  # ensure chronological order

    # Open as xarray dataset
    ds = xr.open_mfdataset(
        tiff_files, combine="nested", concat_dim="time", engine="rasterio"
    )

    # Parse time from filenames (assumes YYYYMMDD.tif naming)
    dates = [os.path.basename(f).split(".")[0] for f in tiff_files]
    ds = ds.assign_coords(time=pd.to_datetime(dates, format="%Y%m%d"))

    # Clean dataset: remove band dim, rename variable
    ds = ds.squeeze("band", drop=True)
    ds = ds.rename({"band_data": "precipitation"})

    # Add CRS
    ds.rio.write_crs("EPSG:4326", inplace=True)

    # Save as NetCDF
    ds.to_netcdf(export_path)

    return ds
