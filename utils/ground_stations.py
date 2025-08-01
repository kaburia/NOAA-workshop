import folium

def plot_stations_folium(dataframes, colors=None):
    """
    Plot stations from one or more dataframes on a Folium map.
    
    Each dataframe must have 'location.latitude' and 'location.longitude' columns.
    'colors' is a list specifying marker colors for each dataframe respectively.
    """
    if colors is None:
        colors = ["blue", "red", "green", "purple", "orange"]
    
    # Create a base map; you can adjust the initial location/zoom as needed
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    # Add markers for each dataframe
    for df, color in zip(dataframes, colors):
        for _, row in df.iterrows():
            folium.Marker(
                location=[row["lat"], row["lon"]],
                tooltip=str(row["station"]),   # <--- Pass the tooltip here
                icon=folium.Icon(color=color)
            ).add_to(m)
    
    return m
