'''
This is a tool to be able access the different weather data sources
easily and also clean the data for easy use
- Ground weather stations
        - TAHMO Data
        - NOAA Data (GHCNd)
- Satellite data
        - CHIRPS data
        - IMERG data
        - TAMSAT Data

- Reanalysis data
        - ERA5 data
        - CBAM data
'''

from setuptools import setup, find_packages

# read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

# read the contents of your requirements file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# tool for the data extraction from the multiple sources
setup(
    name='weather-data-access',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    description='A tool to access and clean weather data from various sources for ML development',
    author='Austin Kaburia',
    author_email='kaburiaaustin1@gmail.com',
    url='https://github.com/kaburia/NOAA-workshop',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'weather-data-access=weather_data_access.cli:main'
        ]
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    )
