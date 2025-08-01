import faostat
import pandas as pd

def get_country_code(country_name):
    """
    Retrieve the FAOSTAT country code for a given country name.
    """
    country_name = country_name.title()
    country = faostat.get_par_df('QCL', 'area')[faostat.get_par_df('QCL', 'area').label.str.contains(country_name)]
    if not country.empty:
        return country.iloc[0]['code']

    raise ValueError(f"Country '{country_name}' not found.")

def get_crop_code(crop_name):
    """
    Retrieve the FAOSTAT item code for a given crop name.
    """
    crop_name = crop_name.title()
    item = faostat.get_par_df('QCL', 'item')[faostat.get_par_df('QCL', 'item').label.str.contains(crop_name)]
    if not item.empty:
        return item.iloc[0]['code']

    raise ValueError(f"Crop '{crop_name}' not found.")

def fetch_yield_data(country_name, crop_name, start_year, end_year, dataset_code='QCL'):
    """
    Fetch yield data for a specific country, crop, and year range.
    """
    try:
        country_code = get_country_code(country_name)
        crop_code = get_crop_code(crop_name)
    except ValueError as e:
        print(e)
        return None

    # Define parameters for the API request
    params = {
        'area': country_code,
        'item': crop_code,
        # 'element': '5419',  # Element code for 'Yield'
        'year': list(range(start_year, end_year + 1))
    }

    # Fetch data
    data = faostat.get_data_df(dataset_code, pars=params, strval=False, null_values=True)

    if data.empty:
        print("No data found for the specified parameters.")
        return None

    # Filter and organize the DataFrame
    data = data[['Area', 'Item', 'Year', 'Value', 'Unit']]
    data = data.rename(columns={'Value': 'Yield'})
    return data
