import ee
import geemap
import xarray as xr
import rioxarray
import os
import glob
import pandas as pd

def get_imerg_raw(start_date, end_date, region=None, export_path="imerg_raw.nc", temporal_agg="raw"):
    """
    Extract IMERG precipitation data from GEE with optional temporal aggregation.

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
    temporal_agg : str
        Temporal aggregation type: "raw" (30-min), "daily", "weekly", or "monthly".

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions (time, y, x) and variable "precipitation" [mm/hr].
    """

    # Load IMERG collection
    imerg = ee.ImageCollection("NASA/GPM_L3/IMERG_V07") \
        .filterDate(start_date, end_date) \
        .select("precipitation")

    if region:
        imerg = imerg.map(lambda img: img.clip(region))

    #  Temporal aggregation logic 
    def aggregate_period(period, unit):
        def agg_func(date):
            start = ee.Date(date)
            end = start.advance(1, unit)
            subset = imerg.filterDate(start, end)
            summed = subset.sum()
            return summed.set('system:time_start', start.millis())
        # Build list of periods
        date_list = ee.List.sequence(0, ee.Date(end_date).difference(ee.Date(start_date), unit).subtract(1))
        return ee.ImageCollection(date_list.map(lambda n: agg_func(ee.Date(start_date).advance(n, unit))))

    if temporal_agg == "daily":
        imerg = aggregate_period("day", "day")
    elif temporal_agg == "weekly":
        imerg = aggregate_period("week", "week")
    elif temporal_agg == "monthly":
        imerg = aggregate_period("month", "month")
    elif temporal_agg != "raw":
        raise ValueError("temporal_agg must be one of: 'raw', 'daily', 'weekly', 'monthly'")

    #  Export directory 
    export_dir = "imerg_raw_temp"
    os.makedirs(export_dir, exist_ok=True)

    #  Export collection 
    geemap.ee_export_image_collection(
        imerg,
        out_dir=export_dir,
        scale=10000,  # ~10 km
        file_per_band=False
    )

    #  Load exported GeoTIFFs 
    tiff_files = glob.glob(os.path.join(export_dir, "*.tif"))
    tiff_files.sort()

    if not tiff_files:
        raise RuntimeError("No IMERG data exported. Check date range, region, or aggregation level.")

    ds = xr.open_mfdataset(
        tiff_files,
        combine="nested",
        concat_dim="time",
        engine="rasterio"
    )

    # Parse time from filenames
    dates = [os.path.basename(f).split('.')[0] for f in tiff_files]
    cleaned_dates = [date[:-2] for date in dates]
    ds["time"] = pd.to_datetime(cleaned_dates, format="%Y%m%d%H%M", errors="coerce")

    # Clean and finalize dataset
    ds = ds.squeeze("band", drop=True)
    ds = ds.rename({"band_data": "precipitation"})
    ds = ds.where(ds != -9999)
    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds.to_netcdf(export_path)

    return ds
