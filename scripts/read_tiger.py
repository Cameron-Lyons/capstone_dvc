import pandas as pd
import geopandas as gpd
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")

VERBOSE = False

def get_tiger_line()->gpd.GeoDataFrame:

    # 2010 Census Shapefiles
    census_2010_tiger_line = [
        'https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_01_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_02_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_04_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_05_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_06_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_08_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_09_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_10_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_11_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_12_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_13_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_15_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_16_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_17_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_18_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_19_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_20_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_21_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_22_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_23_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_24_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_25_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_26_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_27_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_28_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_29_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_30_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_31_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_32_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_33_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_34_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_35_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_36_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_37_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_38_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_39_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_40_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_41_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_42_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_44_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_45_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_46_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_47_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_48_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_49_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_50_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_51_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_53_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_54_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_55_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_56_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_60_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_66_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_69_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_72_tract10.zip','https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/tl_2010_78_tract10.zip'
    ]

    # 2020 Census Shapefiles
    census_2020_tiger_line = [
        'https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_01_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_02_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_04_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_05_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_06_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_08_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_09_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_10_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_11_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_12_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_13_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_15_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_16_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_17_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_18_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_19_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_20_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_21_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_22_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_23_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_24_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_25_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_26_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_27_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_28_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_29_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_30_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_31_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_32_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_33_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_34_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_35_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_36_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_37_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_38_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_39_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_40_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_41_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_42_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_44_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_45_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_46_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_47_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_48_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_49_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_50_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_51_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_53_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_54_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_55_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_56_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_60_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_66_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_69_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_72_tract.zip','https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_78_tract.zip'
    ]

    gdf_union = []

    if VERBOSE:
        for idx, shp in enumerate(tqdm(census_2020_tiger_line)):
            gdf_2020 = gpd.read_file(shp)
            gdf_2010 = gpd.read_file(census_2010_tiger_line[idx])
            gdf_union.append(gpd.overlay(gdf_2010,gdf_2020,how='identity'))
            
        gdf_union = pd.concat(gdf_union)
    else:
        for idx, shp in enumerate(census_2020_tiger_line):
            gdf_2020 = gpd.read_file(shp)
            gdf_2010 = gpd.read_file(census_2010_tiger_line[idx])
            gdf_union.append(gpd.overlay(gdf_2010,gdf_2020,how='identity'))
            
        gdf_union = pd.concat(gdf_union)

    return gdf_union


if __name__ == '__main__':
    """
    Function to retrive data from the Census Bureau
    TIGER/Line dataset

    Requires input of location to store resulting ESRI shapefile
    """

    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_tiger_file', help='Output TIGER/Line file (SHP)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        print('Loading TIGER/Line data...')

    r_df = get_tiger_line()
    if VERBOSE:
        print('Outputting data, please be patient...')
    r_df.to_file(args.output_tiger_file)
