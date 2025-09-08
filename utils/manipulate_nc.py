import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from haversine import haversine
import networkx as nx
import itertools
import haversine as hs

def load_nc_file(filepath):
    """Load a NetCDF file into an xarray.Dataset."""
    return xr.open_dataset(filepath)

def save_to_csv(ds, variable, output_path):
    """Save a variable to CSV with time and station dimensions."""
    df = ds[variable].to_dataframe().reset_index()
    df.to_csv(output_path, index=False)


def list_variables(ds):
    """List available variables in the dataset."""
    return list(ds.data_vars)

def dataset_summary(ds):
    """Print summary stats of the dataset."""
    print(ds)
    print("\nCoordinates:")
    print(ds.coords)
    print("\nAttributes:")
    print(ds.attrs)


def plot_station_variable(ds, station_id, variable, start=None, end=None):
    """Plot a variable's time series for a specific station."""
    da = ds[variable].sel(station=station_id)
    if start and end:
        da = da.sel(time=slice(start, end))
    da.plot.line(x='time', marker='o', figsize=(12, 4))
    plt.title(f"{variable} at station {station_id}")
    plt.grid(True)
    plt.show()


def plot_station_map(ds):
    """Plot station locations on a scatter plot."""
    lats = ds.latitude.values
    lons = ds.longitude.values
    stations = ds.station.values

    plt.figure(figsize=(8, 6))
    plt.scatter(lons, lats, c='red', s=30)
    for i, station in enumerate(stations):
        plt.text(lons[i], lats[i], str(station), fontsize=8)
    plt.title("Station Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()



def aggregate_variable(ds, variable, freq='1D'):
    """Aggregate variable by time across all stations."""
    return ds[variable].resample(time=freq).mean(dim='time')

def aggregate_station_mean(ds, variable):
    """Average value of a variable across all time for each station."""
    return ds[variable].mean(dim='time').to_dataframe().reset_index()



def correlation_matrix(ds, variable):
    """Compute correlation matrix between stations for a variable."""
    df = ds[variable].to_dataframe().unstack(level='station')
    df = df.droplevel(0, axis=1)  # Remove variable level
    return df.corr()

def plot_correlation_matrix(ds, variable):
    """Plot correlation matrix heatmap."""
    corr = correlation_matrix(ds, variable)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title(f"Correlation Matrix for {variable}")
    plt.show()

# select the nearest neighbour (lat,lon) from xarray dataset


def create_proximity_graph(ds, threshold_km):
    """Create graph where edges connect nearby stations."""
    G = nx.Graph()
    stations = ds.station.values

    for station in stations:
        lat = ds.latitude.sel(station=station).item()
        lon = ds.longitude.sel(station=station).item()
        G.add_node(station, latitude=lat, longitude=lon)

    for s1, s2 in itertools.combinations(stations, 2):
        coord1 = (ds.latitude.sel(station=s1).item(), ds.longitude.sel(station=s1).item())
        coord2 = (ds.latitude.sel(station=s2).item(), ds.longitude.sel(station=s2).item())
        dist = haversine(coord1, coord2)
        if dist <= threshold_km:
            G.add_edge(s1, s2, weight=dist)

    return G

def draw_graph(G):
    """Visualize the proximity graph."""
    pos = {n: (G.nodes[n]['longitude'], G.nodes[n]['latitude']) for n in G.nodes}
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=8, edge_color='gray')
    plt.title("Station Proximity Graph")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.show()


def create_neighbor_graph(ds, threshold_km):
        """
        Creates a weighted graph of stations based on geographic proximity.

        Parameters:
        -----------
        ds (xarray.Dataset): Dataset containing station coordinates
        threshold_km (float): Maximum connection distance in kilometers

        Returns:
        -----------
        nx.Graph: NetworkX graph with:
            - Nodes: Station IDs with latitude/longitude attributes
            - Edges: Connections between stations within threshold distance
            - Edge weights: Haversine distances in kilometers
        """
        G = nx.Graph()

        # Add nodes with coordinates
        stations = ds.station.values
        variables = list(ds.data_vars)


        for station in stations:
            try:
                lat = ds.latitude.sel(station=station).item()
                lon = ds.longitude.sel(station=station).item()
                # Extract time-series data for each variable
                var_attrs = {}
                for var in variables:
                    try:
                        series = ds[var].sel(station=station).values
                        var_attrs[var] = series.tolist()  # Convert to native Python list
                    except KeyError:
                        var_attrs[var] = []  # Fallback if variable not available

                G.add_node(station, latitude=lat, longitude=lon, **var_attrs)
            except KeyError:
                print(f"Warning: Missing coordinates for station {station}, skipping")
                continue

        # Create all possible station pairs
        valid_stations = [n for n in G.nodes]
        pairs = list(itertools.combinations(valid_stations, 2))

        # Add edges for nearby stations
        for station1, station2 in pairs:
            coords1 = (G.nodes[station1]['latitude'], G.nodes[station1]['longitude'])
            coords2 = (G.nodes[station2]['latitude'], G.nodes[station2]['longitude'])

            distance = hs.haversine(coords1, coords2)

            if distance <= threshold_km:
                G.add_edge(station1, station2, weight=distance)

        return G

