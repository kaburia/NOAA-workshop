import ee
import geemap
import xarray as xr
import rioxarray
import os
import glob
import pandas as pd

def get_imerg_raw(start_date, end_date, region=None, export_path="imerg_raw.nc"):
    """
    Extract raw IMERG (30-min, mm/hr) precipitation data from GEE into xarray.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD).
    end_date : str
        End date (YYYY-MM-DD).
    region : ee.Geometry, optional
        Region of interest. If None, global data is used.
    export_path : str
        Path to save NetCDF file.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (time, y, x) and variable "precipitation".
        Units: mm/hr at 30-min intervals.
    """

    # Load IMERG collection
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V06") \
        .filterDate(start_date, end_date) \
        .select("precipitationCal")

    if region:
        imerg = imerg.map(lambda img: img.clip(region))

    # Export directory
    export_dir = "imerg_raw_temp"
    os.makedirs(export_dir, exist_ok=True)

    # Export collection
    geemap.ee_export_image_collection(
        imerg,
        out_dir=export_dir,
        scale=10000,  # ~10 km
        file_per_band=False
    )

    # Load GeoTIFFs
    tiff_files = glob.glob(os.path.join(export_dir, "*.tif"))
    tiff_files.sort()

    if not tiff_files:
        raise RuntimeError("No IMERG data exported. Check date range or region.")

    ds = xr.open_mfdataset(
        tiff_files,
        combine="nested",
        concat_dim="time",
        engine="rasterio"
    )

    # Parse times from filenames (they include YYYYMMDDHHMM)
    dates = [os.path.basename(f).split('.')[0] for f in tiff_files]
    # Remove the trailing '00' before parsing
    cleaned_dates = [date[:-2] for date in dates]
    ds["time"] = pd.to_datetime(cleaned_dates, format="%Y%m%d%H%M")

    # Clean up variable names
    ds = ds.squeeze("band", drop=True)
    ds = ds.rename({"band_data": "precipitation"})
    ds = ds.where(ds != -9999)  # mask NoData
    ds.rio.write_crs("EPSG:4326", inplace=True)

    # Save
    ds.to_netcdf(export_path)

    return ds