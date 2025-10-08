import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import xarray as xr
from IPython.display import HTML

def point_plot(
    weather_df,
    metadata_df,
    variable_name="Observation",
    cmap="viridis",
    robust=True,
    fig_title=None,
    interval=300,
    bbox=None,
    save=False,
    metadata_columns=None,
    grid_da=None,
    grid_cmap="coolwarm",
    grid_alpha=0.6,
    grid_da_var=None  
):
    """
    Visualize point-based weather station data and optionally overlay on a gridded Xarray dataset.

    Args:
        weather_df (pd.DataFrame): Time-indexed DataFrame with stations as columns.
        metadata_df (pd.DataFrame): Station metadata with IDs and coordinates.
        variable_name (str): Name of variable being visualized (for point data).
        cmap (str): Colormap for point data.
        robust (bool): Use 2nd–98th percentile limits for normalization.
        fig_title (str): Figure title.
        interval (int): Animation interval in milliseconds.
        bbox (list): [lon_min, lon_max, lat_min, lat_max]. Inferred if None.
        save (bool): Save animation as GIF if True.
        metadata_columns (list): [station_id, lat, lon] column names.
        grid_da (xr.DataArray or xr.Dataset): Optional Xarray grid to overlay. If Dataset, grid_da_var must be specified.
        grid_cmap (str): Colormap for the gridded field.
        grid_alpha (float): Transparency for the gridded field.
        grid_da_var (str): Name of the variable in grid_da if grid_da is a Dataset.

    Returns:
        HTML: Inline animation for Jupyter display.
    """
    # --- Validation and setup ---
    if metadata_columns is None:
        metadata_columns = ["station_id", "lat", "lon"]
    station_col, lat_col, lon_col = metadata_columns

    for col in [station_col, lat_col, lon_col]:
        if col not in metadata_df.columns:
            raise ValueError(f"Missing required metadata column: '{col}' in metadata_df")

    # --- Prepare spatial data ---
    # Ensure weather_df columns match metadata_df station IDs for merging
    # This assumes weather_df columns are the station IDs
    if weather_df.columns.name != station_col:
         weather_df.columns.name = station_col


    lons = metadata_df.set_index(station_col).loc[weather_df.columns][lon_col].values
    lats = metadata_df.set_index(station_col).loc[weather_df.columns][lat_col].values

    # --- Color normalization ---
    data_values = weather_df.values.flatten()
    vmin = np.nanpercentile(data_values, 2 if robust else 0)
    vmax = np.nanpercentile(data_values, 98 if robust else 100)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- Create figure ---
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue")
    if bbox is not None:
        ax.set_extent(bbox)
    else:
         # Set extent based on station coordinates with padding if no bbox is provided
        pad = 0.5  # degrees of padding
        lon_min, lon_max = lons.min() - pad, lons.max() + pad
        lat_min, lat_max = lats.min() - pad, lats.max() + pad
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


    # --- Optional: plot gridded Xarray background ---
    grid_plot = None
    grid_cbar_label = "Grid"
    if grid_da is not None:
        if isinstance(grid_da, xr.Dataset):
            if grid_da_var is None or grid_da_var not in grid_da.data_vars:
                raise ValueError("grid_da_var must be specified and exist in grid_da Dataset")
            grid_data_to_plot = grid_da[grid_da_var]
            grid_cbar_label = grid_da_var if grid_da_var else "Grid"
        elif isinstance(grid_da, xr.DataArray):
            grid_data_to_plot = grid_da
            grid_cbar_label = grid_da.name if grid_da.name else "Grid"
        else:
            raise TypeError("grid_da must be an xarray Dataset or DataArray")


        # Select nearest time step for animation frame 0 if time dimension exists
        initial_grid_frame = grid_data_to_plot.isel(time=0) if "time" in grid_data_to_plot.dims else grid_data_to_plot

        grid_plot = initial_grid_frame.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=grid_cmap,
            alpha=grid_alpha,
            add_colorbar=True,
            cbar_kwargs={"shrink": 0.7, "pad": 0.02, "label": grid_cbar_label},
        )


    # --- Plot station points ---
    scatter = ax.scatter(
        lons,
        lats,
        c=weather_df.iloc[0].values, # Use .values to get numpy array
        cmap=cmap,
        norm=norm,
        s=50,
        transform=ccrs.PlateCarree(),
        edgecolor="black",
        linewidth=0.3,
        zorder=3
    )

    # --- Colorbar for point data---
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label(variable_name, fontsize=12)

    # --- Title ---
    if fig_title is None:
        fig_title = f"{variable_name} Over Time"
    time_index = weather_df.index
    initial_time_label = time_index[0].strftime("%Y-%m-%d %H:%M") if isinstance(time_index[0], pd.Timestamp) else str(time_index[0])
    title = ax.set_title(f"{fig_title}\n{initial_time_label}", fontsize=14)

    # --- Animation update function ---
    def update(frame):
        values = weather_df.iloc[frame].values
        scatter.set_array(values)

        # Update gridded background if it exists and has a time dimension
        if grid_plot is not None and "time" in grid_data_to_plot.dims:
            # Remove previous grid image
            if len(ax.images) > 0:
                 ax.images[-1].remove()

            current_grid_frame = grid_data_to_plot.isel(time=frame)
            current_grid_frame.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=grid_cmap,
                alpha=grid_alpha,
                add_colorbar=False, 
                zorder=1 # Plot grid below points
            )


        current_time_label = time_index[frame].strftime("%Y-%m-%d %H:%M") if isinstance(time_index[frame], pd.Timestamp) else str(time_index[frame])
        title.set_text(f"{fig_title}\n{current_time_label}")
        return [scatter, title] + ax.images # Return all artists that were modified


    ani = animation.FuncAnimation(fig, update, frames=len(weather_df), interval=interval, blit=False)
    plt.close(fig)

    if save:
        ani.save(f"{fig_title}.gif", writer="pillow", fps=3, dpi=150)

    return HTML(ani.to_jshtml())
