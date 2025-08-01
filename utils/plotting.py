import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
from IPython.display import HTML
import xarray as xr
from helpers import get_region_geojson
from IPython.display import display


def select(data: xr.Dataset, variable: str, level: int = None, max_steps: int = None) -> xr.Dataset:
    data = data[variable]
    if "batch" in data.dims:
        data = data.isel(batch=0)
    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))
    if level is not None and "level" in data.coords:
        data = data.sel(level=level)
    return data

def scale(data: xr.Dataset, center: float = None, robust: bool = False) -> tuple:
    # Use 2nd and 98th percentiles if robust, else full range.
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))
    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    cmap = "RdBu_r" if center is not None else "viridis"
    norm = matplotlib.colors.Normalize(vmin, vmax)
    return data, norm, cmap

def plot_data(data_dict: dict, fig_title: str, plot_size: float = 5, robust: bool = False,
              cols: int = 4, bbox: list = None, polygon: list = None) -> HTML:
    """
    Plot the provided xarray data on a map, with the plot extent defined by either a bounding box or a polygon.

    Args:
        data_dict (dict): Dictionary where keys are titles and values are tuples (data, norm, cmap).
        fig_title (str): Title of the figure.
        plot_size (float): Size factor for the figure.
        robust (bool): Whether to use robust scaling for color normalization.
        cols (int): Number of columns in the plot grid.
        bbox (list, optional): Bounding box defined as [lon_min, lon_max, lat_min, lat_max]. If provided, it defines the plot extent.
        polygon (list, optional): A list of [lon, lat] coordinates defining a polygon. If provided (and bbox is not), its bounding box is used for the plot extent.
                                  Additionally, if polygon is provided, it is overlaid on each subplot.

    Returns:
        HTML: The animation as HTML.

    Raises:
        ValueError: If neither bbox nor polygon is provided.
    """
    # Determine the plotting extent.
    if bbox is not None:
        extent = bbox
    elif polygon is not None:
        lons = [coord[0] for coord in polygon]
        lats = [coord[1] for coord in polygon]
        extent = [min(lons), max(lons), min(lats), max(lats)]
    else:
        raise ValueError("Either bbox or polygon must be provided to define the plot extent.")

    # Get first dataset to determine number of time steps.
    first_data = next(iter(data_dict.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d[0].sizes.get("time", 1) for d in data_dict.values())

    cols = min(cols, len(data_dict))
    rows = math.ceil(len(data_dict) / cols)
    # Use constrained_layout to automatically handle spacing.
    figure = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows), constrained_layout=True)
    figure.suptitle(fig_title, fontsize=16)

    images = []
    axes = []
    for i, (title, (plot_data_arr, norm, cmap)) in enumerate(data_dict.items()):
        ax = figure.add_subplot(rows, cols, i + 1, projection=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='white')
        ax.add_feature(cfeature.OCEAN)
        # Overlay the polygon if provided.
        if polygon is not None:
            poly_patch = mpatches.Polygon(polygon, closed=True, facecolor='none',
                                          edgecolor='red', linewidth=2, transform=ccrs.PlateCarree())
            ax.add_patch(poly_patch)
        im = ax.imshow(
            plot_data_arr.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap, transform=ccrs.PlateCarree(), extent=extent)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.01,       # reduced padding
            aspect=16,
            shrink=0.7,     # adjusted shrink
            extend=("both" if robust else "neither"))
        images.append(im)
        axes.append(ax)

    # Precompute slices to avoid repeated slicing in update.
    precomputed = {}
    for key, (da, norm, cmap) in data_dict.items():
        precomputed[key] = [da.isel(time=t, missing_dims="ignore") for t in range(max_steps)]

    def update(frame):
        if "time" in first_data.dims:
            date_str = pd.to_datetime(first_data["time"][frame].item()).strftime('%Y-%m-%d')
            figure.suptitle(f"{fig_title}, {date_str}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        for im, (title, (da, norm, cmap)) in zip(images, data_dict.items()):
            im.set_data(precomputed[title][frame])
        return images

    ani = animation.FuncAnimation(fig=figure, func=update, frames=max_steps, interval=250)
    plt.close(figure.number)
    return HTML(ani.to_jshtml())


# Method to plot the data
def plot_xarray_data(xarray_ds, fig_title, column="total_rainfall",
              plot_size=7, robust=True, bbox=None, region_name=None, maps_key=''):

  if bbox is None:
    geojson = get_region_geojson(region_name, maps_key)
    polygon = geojson['geometry']['coordinates'][0]
  else:
    polygon = None
  # Plot the data
  selected_data = select(xarray_ds, column, level=None, max_steps=None)
  scaled_data, norm, cmap = scale(selected_data, robust=robust)

  # Plotting Dictionary
  data_for_plot = {column: (scaled_data, norm, cmap)}
  fig_title = fig_title

  # Plot and display the animation in the notebook.
  html_anim = plot_data(data_for_plot, fig_title, plot_size=7, robust=True, bbox=bbox, polygon=polygon)
  display(html_anim)



# Set the animation embed limit
plt.rcParams['animation.embed_limit'] = 100  # 100MB

def plot_multiple_data(data_dict: dict, fig_title: str, plot_size: float = 5, robust: bool = False,
                       cols: int = 2, bbox: list = None, polygon: list = None):
    """
    Plot multiple xarray datasets in a grid layout with shared animation controls.
    
    Args:
        data_dict (dict): Dictionary where keys are titles and values are tuples. 
                          The tuple should be (data, norm, cmap) or (data, norm, cmap, custom_bbox).
        fig_title (str): Main figure title.
        plot_size (float): Base size for plot elements.
        robust (bool): Whether to use robust scaling.
        cols (int): Maximum number of columns in the grid.
        bbox (list): Global bounding box [lon_min, lon_max, lat_min, lat_max] for subplots.
        polygon (list): Global polygon coordinates for overlay (if bbox is not provided).
    
    Returns:
        tuple: (ani, HTML) where ani is the FuncAnimation object and HTML is its HTML representation.
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib import animation
    from IPython.display import HTML
    import pandas as pd

    # Calculate grid layout
    num_plots = len(data_dict)
    cols = min(cols, num_plots)
    rows = math.ceil(num_plots / cols)
    
    # Optional adjustment for specific numbers of plots
    if num_plots in [5, 7]:
        rows = math.ceil(num_plots / (cols - 1))
        cols -= 1

    # Create figure with custom spacing
    fig = plt.figure(figsize=(plot_size * 2 * cols, plot_size * rows), constrained_layout=False)
    fig.suptitle(fig_title, fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, wspace=0.05, hspace=0.05)

    images = []
    axes = []
    precomputed = {}
    max_steps = 1

    # Loop through each subplot
    for idx, (title, data_tuple) in enumerate(data_dict.items()):
        # Support both 3-item and 4-item tuples.
        if len(data_tuple) == 3:
            data, norm, cmap = data_tuple
            custom_bbox = None
        elif len(data_tuple) == 4:
            data, norm, cmap, custom_bbox = data_tuple
        else:
            raise ValueError("Data tuple must be (data, norm, cmap) or (data, norm, cmap, custom_bbox).")
        
        # Determine extent: per subplot if custom_bbox provided; otherwise use global bbox or polygon.
        if custom_bbox is not None:
            extent = custom_bbox  # Expected format: [lon_min, lon_max, lat_min, lat_max]
        elif bbox is not None:
            extent = bbox
        elif polygon is not None:
            lons = [coord[0] for coord in polygon]
            lats = [coord[1] for coord in polygon]
            extent = [min(lons), max(lons), min(lats), max(lats)]
        else:
            raise ValueError("Either a global bbox, polygon, or per-subplot custom bbox must be provided")

        # Create the subplot
        ax = fig.add_subplot(rows, cols, idx + 1, projection=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, edgecolor='white')
        ax.add_feature(cfeature.OCEAN)
        
        # Optionally add a global polygon overlay if provided.
        if polygon is not None:
            poly_patch = mpatches.Polygon(polygon, closed=True, facecolor='none',
                                          edgecolor='red', linewidth=2, transform=ccrs.PlateCarree())
            ax.add_patch(poly_patch)

        # Precompute time steps for the animation
        time_steps = data.sizes.get("time", 1)
        max_steps = max(max_steps, time_steps)
        precomputed[title] = [data.isel(time=t, missing_dims="ignore") if time_steps > 1 else data
                              for t in range(time_steps)]

        im = ax.imshow(
            precomputed[title][0],
            norm=norm,
            origin="lower",
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extent=extent
        )
        
        # Add colorbar with reduced padding
        cbar = plt.colorbar(
            im, ax=ax, orientation="vertical", pad=0.03,
            aspect=16, shrink=0.7, extend=("both" if robust else "neither")
        )
        cbar.ax.tick_params(labelsize=8)
        
        images.append(im)
        axes.append(ax)

    # Define the animation update function
    def update(frame):
        time_str = ""
        for idx, (title, data_tuple) in enumerate(data_dict.items()):
            if len(data_tuple) == 3:
                data, _, _ = data_tuple
            elif len(data_tuple) == 4:
                data, _, _, _ = data_tuple
            time_steps = data.sizes.get("time", 1)
            if time_steps > 1:
                current_frame = min(frame, time_steps - 1)
                images[idx].set_data(precomputed[title][current_frame])
                time_str = pd.to_datetime(data["time"][current_frame].item()).strftime('%Y-%m-%d')
        if time_str:
            fig.suptitle(f"{fig_title}\n{time_str}", fontsize=16, y=0.98)
        return images

    # Create the animation object
    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=max_steps,
        interval=250,
        blit=True
    )
    
    plt.close(fig)
    return ani, HTML(ani.to_jshtml())


# Updated plotting method that supports multiple variables
def plot_xarray_data2(xarray_ds, fig_title, columns=["total_rainfall"], 
                      plot_size=7, robust=True, bbox=None, region_name=None, 
                      maps_key='', save=False):
    """
    Plot selected variables from an xarray Dataset.
    
    Args:
      xarray_ds: The xarray Dataset containing the variables.
      fig_title (str): Main title for the figure.
      columns (list): List of variable names to plot.
      plot_size (float): Base plot size.
      robust (bool): Use robust scaling if True.
      bbox (list): Bounding box [lon_min, lon_max, lat_min, lat_max].
      region_name (str): Region name for fetching geojson if bbox is None.
        maps_key (str): API key for fetching geojson.
        save (bool): Save the plot as a GIF file.
    """
    if bbox is None and region_name is not None:
        geojson = get_region_geojson(region_name, maps_key)
        polygon = geojson['geometry']['coordinates'][0]
    else:
        polygon = None

    data_for_plot = {}
    for col in columns:
        # 'select' should extract the DataArray for the given column.
        selected_data = select(xarray_ds, col, level=None, max_steps=None)
        # 'scale' should return (scaled_data, norm, cmap) for plotting.
        scaled_data, norm, cmap = scale(selected_data, robust=robust)
        data_for_plot[col] = (scaled_data, norm, cmap)
    
    ani, html_anim = plot_multiple_data(
        data_for_plot, 
        fig_title,
        plot_size=plot_size,
        robust=robust,
        bbox=bbox,
        polygon=polygon
    )

    # Save the animation as a GIF if requested.
    if save:
        ani.save(f"{fig_title}.gif", writer="pillow",
                 fps=4, dpi=200, savefig_kwargs={"facecolor": "white"})
    
    display(html_anim)

    

def compare_xarray_datasets(datasets: list, labels: list, fig_title: str, 
                            plot_size: float = 5, robust: bool = False, 
                            cols: int = 2, 
                            bboxes: list = None,   # NEW: list of bounding boxes
                            polygon: list = None,
                            save: bool = False) -> HTML:
    """
    Compare multiple xarray datasets (each containing a single variable) from the same or different regions
    by plotting them side-by-side in an animated grid.

    Args:
        datasets (list): List of xarray.Dataset objects, each with one variable column.
        labels (list): List of labels corresponding to each dataset.
        fig_title (str): Title for the overall figure.
        plot_size (float): Base size for plot elements.
        robust (bool): Whether to use robust scaling.
        cols (int): Maximum number of columns in the grid.
        bboxes (list): A list of bounding boxes, each defined as [lon_min, lon_max, lat_min, lat_max].
                       The length of bboxes must match the length of datasets.
        polygon (list): Polygon coordinates for overlay if using one region for all datasets.

    Returns:
        HTML: The animated plot as HTML.
    """
    # Prepare a dictionary mapping label -> (scaled_data, norm, cmap, custom_bbox)
    data_for_plot = {}

    # If multiple bboxes are provided, make sure there's one for each dataset.
    if bboxes is not None:
        if len(bboxes) != len(datasets):
            raise ValueError("Length of bboxes must match the number of datasets.")
    else:
        # If no bboxes are given, we'll rely on a single polygon or single bbox in plot_multiple_data
        # (the usual code path).
        bboxes = [None] * len(datasets)

    for (ds, label), bbox_for_ds in zip(zip(datasets, labels), bboxes):
        # Expect each dataset to have exactly one variable.
        var_names = list(ds.data_vars)
        if len(var_names) != 1:
            raise ValueError(f"Dataset with label '{label}' has {len(var_names)} variables; expected exactly one.")
        var_name = var_names[0]
        data_array = ds[var_name]

        # Scale for plotting (robust color normalization, etc.).
        scaled_data, norm, cmap = scale(data_array, robust=robust)

        # Store everything, including a possible custom bbox for this dataset
        data_for_plot[label] = (scaled_data, norm, cmap, bbox_for_ds)

    # Use the modified plot function that supports per-subplot bounding boxes
    ani, html_anim = plot_multiple_data(
        data_for_plot,
        fig_title,
        plot_size=plot_size,
        robust=robust,
        cols=cols,
        bbox=None,       # We pass None here because we'll handle bboxes per subplot
        polygon=polygon
    )

    # Save the animation as a GIF if requested.
    if save:
        ani.save(f"{fig_title}.gif", writer="pillow",
                       fps=4, dpi=400, savefig_kwargs={"facecolor": "white"})
    
    return html_anim


def compare_xarray_datasets2(datasets: list, labels: list, fig_title: str, 
                            plot_size: float = 5, robust: bool = False, 
                            cols: int = 2, 
                            bboxes: list = None,   # NEW: list of bounding boxes
                            polygon: list = None,
                            save: bool = False) -> HTML:
    """
    Compare multiple xarray datasets (each containing a single variable) from the same or different regions
    by plotting them side-by-side in an animated grid.

    Args:
        datasets (list): List of xarray.Dataset objects, each with one variable column.
        labels (list): List of labels corresponding to each dataset.
        fig_title (str): Title for the overall figure.
        plot_size (float): Base size for plot elements.
        robust (bool): Whether to use robust scaling.
        cols (int): Maximum number of columns in the grid.
        bboxes (list): A list of bounding boxes, each defined as [lon_min, lon_max, lat_min, lat_max].
                       The length of bboxes must match the length of datasets.
        polygon (list): Polygon coordinates for overlay if using one region for all datasets.

    Returns:
        HTML: The animated plot as HTML.
    """
    # Prepare a dictionary mapping label -> (scaled_data, norm, cmap, custom_bbox)
    data_for_plot = {}

    # If multiple bboxes are provided, make sure there's one for each dataset.
    if bboxes is not None:
        if len(bboxes) != len(datasets):
            raise ValueError("Length of bboxes must match the number of datasets.")
    else:
        # If no bboxes are given, we'll rely on a single polygon or single bbox in plot_multiple_data
        # (the usual code path).
        bboxes = [None] * len(datasets)

    for (ds, label), bbox_for_ds in zip(zip(datasets, labels), bboxes):
        # Expect each dataset to have exactly one variable.
        var_names = list(ds.data_vars)
        if len(var_names) != 1:
            raise ValueError(f"Dataset with label '{label}' has {len(var_names)} variables; expected exactly one.")
        var_name = var_names[0]
        data_array = ds[var_name]

        # Scale for plotting (robust color normalization, etc.).
        scaled_data, norm, cmap = scale(data_array, robust=robust)

        # Store everything, including a possible custom bbox for this dataset
        data_for_plot[label] = (scaled_data, norm, cmap, bbox_for_ds)

    # Use the modified plot function that supports per-subplot bounding boxes
    ani, html_anim = plot_multiple_data(
        data_for_plot,
        fig_title,
        plot_size=plot_size,
        robust=robust,
        cols=cols,
        bbox=None,       # We pass None here because we'll handle bboxes per subplot
        polygon=polygon
    )

    # Save the animation as a GIF if requested.
    if save:
        ani.save(f"{fig_title}.gif", writer="pillow",
                       fps=4, dpi=400, savefig_kwargs={"facecolor": "white"})
    
    return html_anim




