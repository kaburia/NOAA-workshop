import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

curr_dir = os.getcwd()
root_dir = str(Path(curr_dir).parent)
print(f"Root dir: {root_dir}")
print(f"Curr dir: {curr_dir}")

# Append path
sys.path.append(os.path.join(curr_dir, 'get-station-data'))
from get_station_data.util import nearest_stn

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

def subset_noaa_stations_by_country(stn_md, country_code):
    return stn_md[stn_md['station'].str.startswith(country_code)]

# given the variable/element return the dataframe
def subset_weather_data_by_variable(df, var, pivot=False, qflag=False):
  df = df[ df['element'] == var ]

  ### Tidy up columns
  df = df.rename(index=str, columns={"value": var})
  df = df.drop(['element'], axis=1)
  if pivot:
    return df.pivot(columns='station', values=var, index='date')
  elif qflag:
    df_flags = df[['station', 'qflag', 'date']].pivot(columns='station',
                                                      values='qflag',
                                                      index='date')
    return df_flags

  else:
    return df

def plot_station_data(station_id, df_pr, df_tavg, df_tmin, df_tmax):
    """Plots precipitation, average temperature, minimum temperature, and maximum temperature for a given station.

    Args:
        station_id: The ID of the station to plot data for.
        df_pr: DataFrame containing precipitation data.
        df_tavg: DataFrame containing average temperature data.
        df_tmin: DataFrame containing minimum temperature data.
        df_tmax: DataFrame containing maximum temperature data.
    """

    fig, axes = plt.subplots(4, 1, figsize=(15, 7), sharex=True)
    fig.suptitle(f"Weather Data for Station: {station_id}")

    # Plot Precipitation
    axes[0].bar(df_pr.index, df_pr[station_id], label='Precipitation (PRCP)')
    axes[0].set_ylabel('Precipitation (mm)')
    axes[0].set_ylim(0, 60)
    axes[0].legend()

    # Plot Average Temperature
    axes[2].plot(df_tavg.index, df_tavg[station_id], label='Average Temperature (TAVG)')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].legend()

    # Plot Minimum Temperature
    axes[1].plot(df_tmin.index, df_tmin[station_id], label='Minimum Temperature (TMIN)')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend()

    # Plot Maximum Temperature
    axes[3].plot(df_tmax.index, df_tmax[station_id], label='Maximum Temperature (TMAX)')
    axes[3].set_ylabel('Temperature (°C)')
    axes[3].legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def get_nearest_wmo_station(stn_md, tahmo_station_id, tahmo_metadata, neighbours=1):
    lat, lon = tahmo_metadata[tahmo_metadata.station == tahmo_station_id][['lat', 'lon']].values[0]
    return nearest_stn(stn_md, lon, lat, n_neighbours=neighbours).station.values[0]

