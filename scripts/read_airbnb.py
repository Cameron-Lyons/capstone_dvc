import gzip
import os

import urllib.request
from io import BytesIO
from tqdm import tqdm
import requests

import pandas as pd
import numpy as np
import geopandas as gpd

VERBOSE = False

def get_airbnb_data()->gpd.GeoDataFrame:
    data_files = [
        'http://data.insideairbnb.com/united-states/nc/asheville/2021-09-16/data/listings.csv.gz','http://data.insideairbnb.com/united-states/tx/austin/2021-09-14/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ma/boston/2021-09-19/data/listings.csv.gz','http://data.insideairbnb.com/united-states/fl/broward-county/2021-09-27/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ma/cambridge/2021-09-29/data/listings.csv.gz','http://data.insideairbnb.com/united-states/il/chicago/2021-09-16/data/listings.csv.gz','http://data.insideairbnb.com/united-states/nv/clark-county-nv/2021-09-21/data/listings.csv.gz','http://data.insideairbnb.com/united-states/oh/columbus/2021-09-27/data/listings.csv.gz','http://data.insideairbnb.com/united-states/co/denver/2021-09-30/data/listings.csv.gz','http://data.insideairbnb.com/united-states/hi/hawaii/2021-09-12/data/listings.csv.gz','http://data.insideairbnb.com/united-states/nj/jersey-city/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/los-angeles/2021-09-08/data/listings.csv.gz','http://data.insideairbnb.com/united-states/tn/nashville/2021-09-21/data/listings.csv.gz','http://data.insideairbnb.com/united-states/la/new-orleans/2021-09-08/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ny/new-york-city/2021-09-01/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/oakland/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/pacific-grove/2021-09-30/data/listings.csv.gz','http://data.insideairbnb.com/united-states/or/portland/2021-09-24/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ri/rhode-island/2021-09-30/data/listings.csv.gz','http://data.insideairbnb.com/united-states/or/salem-or/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/san-diego/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/san-francisco/2021-10-06/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/san-mateo-county/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/santa-clara-county/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/ca/santa-cruz-county/2021-09-30/data/listings.csv.gz','http://data.insideairbnb.com/united-states/wa/seattle/2021-09-25/data/listings.csv.gz','http://data.insideairbnb.com/united-states/mn/twin-cities-msa/2021-09-24/data/listings.csv.gz','http://data.insideairbnb.com/united-states/dc/washington-dc/2021-09-16/data/listings.csv.gz'
    ]

    g_df = []
    
    if VERBOSE:
        for filename in tqdm(data_files):
            url = urllib.request.urlopen(filename)
            with gzip.open(BytesIO(url.read()), 'rb') as l_file:
                l_df = pd.read_csv(l_file)
                g_df.append(l_df)
    else:
        for filename in data_files:
            url = urllib.request.urlopen(filename)
            with gzip.open(BytesIO(url.read()), 'rb') as l_file:
                l_df = pd.read_csv(l_file)
                g_df.append(l_df)

    g_df = pd.concat(g_df)

    g_df = gpd.GeoDataFrame(
        g_df, 
        geometry=gpd.points_from_xy(
            g_df['longitude'], 
            g_df['latitude']),
        crs='EPSG:4326'
    )

    return g_df

if __name__ == '__main__':
    """
    Function to retrive data from the inside Airbnb dataset

    Requires input of location to store resulting GeoJSON file
    """

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_airbnb_file', help='Output Airbnb file (GeoJSON)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        print('Loading Airbnb data...')

    r_df = get_airbnb_data()

    r_df.to_file(filename=args.output_airbnb_file, driver='GeoJSON')
