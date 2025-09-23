import pandas as pd
import matplotlib.pyplot as plt


# Check for flatlines when values do not change for k days (k=5) with a rolling window
def detect_flatlines(data, window_size=5):
    # store the station and the window in a dictionary
    flatline_info = {}
    for station in data.columns:
        rolling_std = data[station].rolling(window=window_size).std()
        flatline_periods = rolling_std[rolling_std == 0].index.tolist()
        if flatline_periods:
            flatline_info[station] = flatline_periods
    return flatline_info

# plot the flatline stations highlighting flatline periods in red
def plot_flatline_stations(data, flatline_info, window_size=5, random=True):
    import random

    if not flatline_info:
        print("No flatlines detected.")
        return

    station = random.choice(list(flatline_info.keys())) if random else list(flatline_info.keys())[0]
    station_data = data[station].copy()

    # Build a mask for flatline periods
    mask = pd.Series(False, index=station_data.index)
    periods = flatline_info[station]
    for end_ts in periods:
        if end_ts in station_data.index:
            # mark the window contributing to zero std
            start_pos = station_data.index.get_loc(end_ts) - window_size + 1
            if start_pos >= 0:
                flat_index = station_data.index[start_pos: station_data.index.get_loc(end_ts) + 1]
                mask.loc[flat_index] = True

    plt.figure(figsize=(10, 5))
    plt.plot(station_data.index, station_data.values, color='steelblue', marker='o', label='Values')

    # Overlay flatline segments in red
    if mask.any():
        flat_series = station_data[mask]
        # To keep contiguous red segments joined, plot as line over masked points sorted
        plt.plot(flat_series.index, flat_series.values, color='red', linewidth=2.5, marker='o', label='Flatline')

    plt.title(f'Precipitation Data for Station {station} (Flatlines Highlighted)')
    plt.xlabel('Date')
    plt.ylabel('Precipitation (mm)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Station Code: {station}")
    print(f"Data Range: {station_data.min()} mm to {station_data.max()} mm")
    print(f"Number of Records: {len(station_data)}")
    if periods:
        print(f"Flatline window end timestamps (window={window_size}):")
        for ts in periods:
            print(f"  - {ts}")
    else:
        print("No flatline periods for this station.")

