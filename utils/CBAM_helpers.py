import base64
import json
import requests
import pandas as pd
import io
import xarray as xr

from helpers import df_to_xarray2

class CBAMClient:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.base_url = "https://tngap.sta.do.kartoza.com/api/v1/measurement/"
        self.token = None

    def authenticate(self):
        username = self.config['username']
        password = self.config['password']
        token_str = f"{username}:{password}"
        token_bytes = token_str.encode('utf-8')
        self.token = base64.b64encode(token_bytes).decode('utf-8')
        return self.token

    def get_data(self, start_date, end_date, attributes, bbox=None, 
                 location_name=None, lat=None, lon=None, 
                 product="cbam_historical_analysis", output_type='csv'):
        if not self.token:
            self.authenticate()

        # Convert attributes list to comma-separated string if needed.
        attr_str = ",".join(attributes) if isinstance(attributes, list) else attributes

        params = {
            "product": product,
            "attributes": attr_str,
            "start_date": start_date,
            "end_date": end_date,
            "output_type": output_type,
        }
        if bbox:
            bbox = ",".join(map(str, bbox))
            params["bbox"] = bbox
            # print(params["bbox"])
        if location_name:
            # If location_name is a list (e.g., polygon coordinates), join them into a single comma-separated string.
            if isinstance(location_name, list):
                location_name = ",".join(map(str, location_name))
            params["location_name"] = location_name
        if lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon

        headers = {
            'accept': 'application/json',
            'authorization': f'Basic {self.token}',
        }

        response = requests.get(self.base_url, params=params, headers=headers)
        response.raise_for_status()

        # For netcdf, return binary content; for other formats, return text.
        if output_type == 'netcdf':
            return response.content
        else:
            return response.text

def extract_cbam_data(start_date, end_date, attributes, cbam_client, 
                      output_type='csv', df=False, bbox=None, 
                      lat=None, lon=None, location_name=None):
    """
    Extract CBAM data based on provided parameters.

    Parameters:
      - start_date: Start date in 'YYYY-MM-DD' format.
      - end_date: End date in 'YYYY-MM-DD' format.
      - attributes: List of attributes to fetch (e.g., ["max_temperature", "min_temperature"]).
      - cbam_client: Instance of CBAMClient.
      - output_type: 'csv', 'netcdf', or 'json'. Default is 'netcdf'.
      - df: If True and output_type is 'csv', return a pandas DataFrame instead of an xarray Dataset.
      - bbox: Bounding box string (e.g., '33.9, -4.67, 41.89, 5.5').
      - lat: Latitude for a single point query.
      - lon: Longitude for a single point query.
      - location_name: For polygon/bounding box queries, can be a shapefile name or a list of coordinates.
    """
    data = cbam_client.get_data(
        start_date, 
        end_date, 
        attributes, 
        bbox=bbox, 
        lat=lat, 
        lon=lon, 
        location_name=location_name,
        output_type=output_type
    )
    
    if output_type == 'csv':
        # Convert CSV text into a DataFrame.
        cbam_df = pd.read_csv(io.StringIO(data))
        cbam_df['date'] = pd.to_datetime(cbam_df['date'])
        cbam_df = cbam_df.sort_values('date')
        # print(cbam_df.head())
        if df:
            return cbam_df
        else:
            # Convert the DataFrame into an xarray Dataset.
            return df_to_xarray2(cbam_df, variables=attributes)
    elif output_type == 'netcdf':
        # Open netCDF data using xarray from binary content.
        ds = xr.open_dataset(io.BytesIO(data))
        return ds
    elif output_type == 'json':
        # Parse and return JSON data.
        return json.loads(data)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
