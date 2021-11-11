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
API_KEY = '80bf22e058fc4f1f34f51638c75678386a48a393'
DATA_2020 = ['H1_001N','H1_002N','H1_003N','P1_001N','P1_002N','P1_003N','P1_004N','P1_005N','P1_006N','P1_007N','P1_008N','P1_009N','P1_010N','P1_011N','P1_012N','P1_013N','P1_014N','P1_015N','P1_016N','P1_017N','P1_018N','P1_019N','P1_020N','P1_021N','P1_022N','P1_023N','P1_024N','P1_025N','P1_026N','P1_027N','P1_028N','P1_029N','P1_030N','P1_031N','P1_032N','P1_033N','P1_034N','P1_035N','P1_036N','P1_037N','P1_038N','P1_039N','P1_040N','P1_041N','P1_042N','P1_043N','P1_044N','P1_045N','P1_046N','P1_047N','P1_048N','P1_049N','P1_050N','P1_051N','P1_052N','P1_053N','P1_054N','P1_055N','P1_056N','P1_057N','P1_058N','P1_059N','P1_060N','P1_061N','P1_062N','P1_063N','P1_064N','P1_065N','P1_066N','P1_067N','P1_068N','P1_069N','P1_070N','P1_071N','P2_001N','P2_002N','P2_003N','P2_004N','P2_005N','P2_006N','P2_007N','P2_008N','P2_009N','P2_010N','P2_011N','P2_012N','P2_013N','P2_014N','P2_015N','P2_016N','P2_017N','P2_018N','P2_019N','P2_020N','P2_021N','P2_022N','P2_023N','P2_024N','P2_025N','P2_026N','P2_027N','P2_028N','P2_029N','P2_030N','P2_031N','P2_032N','P2_033N','P2_034N','P2_035N','P2_036N','P2_037N','P2_038N','P2_039N','P2_040N','P2_041N','P2_042N','P2_043N','P2_044N','P2_045N','P2_046N','P2_047N','P2_048N','P2_049N','P2_050N','P2_051N','P2_052N','P2_053N','P2_054N','P2_055N','P2_056N','P2_057N','P2_058N','P2_059N','P2_060N','P2_061N','P2_062N','P2_063N','P2_064N','P2_065N','P2_066N','P2_067N','P2_068N','P2_069N','P2_070N','P2_071N','P2_072N','P2_073N','P3_001N','P3_002N','P3_003N','P3_004N','P3_005N','P3_006N','P3_007N','P3_008N','P3_009N','P3_010N','P3_011N','P3_012N','P3_013N','P3_014N','P3_015N','P3_016N','P3_017N','P3_018N','P3_019N','P3_020N','P3_021N','P3_022N','P3_023N','P3_024N','P3_025N','P3_026N','P3_027N','P3_028N','P3_029N','P3_030N','P3_031N','P3_032N','P3_033N','P3_034N','P3_035N','P3_036N','P3_037N','P3_038N','P3_039N','P3_040N','P3_041N','P3_042N','P3_043N','P3_044N','P3_045N','P3_046N','P3_047N','P3_048N','P3_049N','P3_050N','P3_051N','P3_052N','P3_053N','P3_054N','P3_055N','P3_056N','P3_057N','P3_058N','P3_059N','P3_060N','P3_061N','P3_062N','P3_063N','P3_064N','P3_065N','P3_066N','P3_067N','P3_068N','P3_069N','P3_070N','P3_071N','P4_001N','P4_002N','P4_003N','P4_004N','P4_005N','P4_006N','P4_007N','P4_008N','P4_009N','P4_010N','P4_011N','P4_012N','P4_013N','P4_014N','P4_015N','P4_016N','P4_017N','P4_018N','P4_019N','P4_020N','P4_021N','P4_022N','P4_023N','P4_024N','P4_025N','P4_026N','P4_027N','P4_028N','P4_029N','P4_030N','P4_031N','P4_032N','P4_033N','P4_034N','P4_035N','P4_036N','P4_037N','P4_038N','P4_039N','P4_040N','P4_041N','P4_042N','P4_043N','P4_044N','P4_045N','P4_046N','P4_047N','P4_048N','P4_049N','P4_050N','P4_051N','P4_052N','P4_053N','P4_054N','P4_055N','P4_056N','P4_057N','P4_058N','P4_059N','P4_060N','P4_061N','P4_062N','P4_063N','P4_064N','P4_065N','P4_066N','P4_067N','P4_068N','P4_069N','P4_070N','P4_071N','P4_072N','P4_073N','P5_001N','P5_002N','P5_003N','P5_004N','P5_005N','P5_006N','P5_007N','P5_008N','P5_009N','P5_010N']
DATA_2010 = ['H001001','H001002','H001003','P001001','P001002','P001003','P001004','P001005','P001006','P001007','P001008','P001009','P001010','P001011','P001012','P001013','P001014','P001015','P001016','P001017','P001018','P001019','P001020','P001021','P001022','P001023','P001024','P001025','P001026','P001027','P001028','P001029','P001030','P001031','P001032','P001033','P001034','P001035','P001036','P001037','P001038','P001039','P001040','P001041','P001042','P001043','P001044','P001045','P001046','P001047','P001048','P001049','P001050','P001051','P001052','P001053','P001054','P001055','P001056','P001057','P001058','P001059','P001060','P001061','P001062','P001063','P001064','P001065','P001066','P001067','P001068','P001069','P001070','P001071','P002001','P002002','P002003','P002004','P002005','P002006','P002007','P002008','P002009','P002010','P002011','P002012','P002013','P002014','P002015','P002016','P002017','P002018','P002019','P002020','P002021','P002022','P002023','P002024','P002025','P002026','P002027','P002028','P002029','P002030','P002031','P002032','P002033','P002034','P002035','P002036','P002037','P002038','P002039','P002040','P002041','P002042','P002043','P002044','P002045','P002046','P002047','P002048','P002049','P002050','P002051','P002052','P002053','P002054','P002055','P002056','P002057','P002058','P002059','P002060','P002061','P002062','P002063','P002064','P002065','P002066','P002067','P002068','P002069','P002070','P002071','P002072','P002073','P003001','P003002','P003003','P003004','P003005','P003006','P003007','P003008','P003009','P003010','P003011','P003012','P003013','P003014','P003015','P003016','P003017','P003018','P003019','P003020','P003021','P003022','P003023','P003024','P003025','P003026','P003027','P003028','P003029','P003030','P003031','P003032','P003033','P003034','P003035','P003036','P003037','P003038','P003039','P003040','P003041','P003042','P003043','P003044','P003045','P003046','P003047','P003048','P003049','P003050','P003051','P003052','P003053','P003054','P003055','P003056','P003057','P003058','P003059','P003060','P003061','P003062','P003063','P003064','P003065','P003066','P003067','P003068','P003069','P003070','P003071','P004001','P004002','P004003','P004004','P004005','P004006','P004007','P004008','P004009','P004010','P004011','P004012','P004013','P004014','P004015','P004016','P004017','P004018','P004019','P004020','P004021','P004022','P004023','P004024','P004025','P004026','P004027','P004028','P004029','P004030','P004031','P004032','P004033','P004034','P004035','P004036','P004037','P004038','P004039','P004040','P004041','P004042','P004043','P004044','P004045','P004046','P004047','P004048','P004049','P004050','P004051','P004052','P004053','P004054','P004055','P004056','P004057','P004058','P004059','P004060','P004061','P004062','P004063','P004064','P004065','P004066','P004067','P004068','P004069','P004070','P004071','P004072','P004073']

def combine_files(census_file:str,airbnb_file:str)->gpd.GeoDataFrame:
    """
    First step in processing performs
    a spatial join of the Census and
    Airbnb files.
    """
    census_df = gpd.read_file(census_file)
    airbnb_df = gpd.read_file(airbnb_file)
    census_df  = census_df.to_crs('EPSG:4326')
    used_tl = gpd.sjoin(census_df,airbnb_df)

    return used_tl

def query_census_2020(used_tracts_df:gpd.GeoDataFrame)->pd.DataFrame:
    """
    Queries census bureau API for 2020 data
    """
    
    l_df = used_tracts_df.groupby(['STATEFP','COUNTYFP']).count().reset_index()
    n_step = 50

    df_2020 = []
    for idx, row in tqdm(l_df.iterrows(), total=l_df.shape[0]):
        l_statefp = row['STATEFP']
        l_county = row['COUNTYFP']
        l_tracts = used_tracts_df[(used_tracts_df['STATEFP']==l_statefp) &(used_tracts_df['COUNTYFP']==l_county)]['TRACTCE'].unique()
        l_tracts = ",".join(l_tracts)
        last_n = -n_step
        next_n = last_n+n_step
        iter_df = []
        while next_n < len(DATA_2020):
            if last_n+n_step*2 > len(DATA_2020):
                last_n += n_step
                next_n = len(DATA_2020)
            else:
                last_n += n_step
                next_n = last_n+n_step
            data_step = ",".join(DATA_2020[last_n:next_n])
            census_url = f'https://api.census.gov/data/2020/dec/pl?get={data_step}&for=tract:{l_tracts}&in=state:{l_statefp}%20county:{l_county}&key={API_KEY}'
            
            try:
                response=requests.get(census_url)
                data=response.json()
                i_df=pd.DataFrame(data[1:], columns=data[0])
                #print(len(iter_df))
                if len(iter_df)==0:
                    iter_df = i_df
                else:
                    iter_df = pd.concat([iter_df,i_df], axis=1)
            except:
                pass
                #print(response)
        if len(iter_df) > 0:
            if len(df_2020) > 0:
                df_2020 = pd.concat([df_2020,iter_df])
            else:
                df_2020 = iter_df
        
    #df_2020 = pd.concat(df_2020)
    return df_2020

def query_census_2010(used_tracts_df:gpd.GeoDataFrame)->pd.DataFrame:
    """
    Queries census bureau API for 2010 data
    """
    l_df = used_tracts_df.groupby(['STATEFP10','COUNTYFP10']).count().reset_index()
    n_step = 50

    df_2010 = []
    for idx, row in tqdm(l_df.iterrows(), total=l_df.shape[0]):
        l_statefp = row['STATEFP10']
        l_county = row['COUNTYFP10']
        l_tracts = used_tracts_df[(used_tracts_df['STATEFP10']==l_statefp) &(used_tracts_df['COUNTYFP10']==l_county)]['TRACTCE10'].unique()
        l_tracts = ",".join(l_tracts)
        last_n = -n_step
        next_n = last_n+n_step
        iter_df = []
        while next_n < len(DATA_2010):
            if last_n+n_step*2 > len(DATA_2010):
                last_n += n_step
                next_n = len(DATA_2010)
            else:
                last_n += n_step
                next_n = last_n+n_step
            data_step = ",".join(DATA_2010[last_n:next_n])
            #https://api.census.gov/data/2010/dec/pl?get=P001001,NAME&for=tract:020500&in=state:01%20county:001&key=YOUR_KEY_GOES_HERE
            census_url = f'https://api.census.gov/data/2010/dec/pl?get={data_step}&for=tract:{l_tracts}&in=state:{l_statefp}%20county:{l_county}&key={API_KEY}'
            
            try:
                response=requests.get(census_url)
                data=response.json()
                i_df=pd.DataFrame(data[1:], columns=data[0])
                #print(len(iter_df))
                if len(iter_df)==0:
                    iter_df = i_df
                else:
                    iter_df = pd.concat([iter_df,i_df], axis=1)
            except:
                pass
                #print(response)
        if len(iter_df) > 0:
            if len(df_2010) > 0:
                df_2010 = pd.concat([df_2010,iter_df])
            else:
                df_2010 = iter_df
        
    return df_2010

def generate_final(used_tracts_df:gpd.GeoDataFrame,df_2010:pd.DataFrame,df_2020:pd.DataFrame)->pd.DataFrame:
    df_2010 = df_2010.loc[:,~df_2010.columns.duplicated()]
    df_2020 = df_2020.loc[:,~df_2020.columns.duplicated()]
    used_tracts_df = used_tracts_df.merge(df_2010, left_on=['STATEFP10','COUNTYFP10','TRACTCE10'], right_on=['state','county','tract'])
    used_tracts_df = used_tracts_df.merge(df_2020, left_on=['STATEFP','COUNTYFP','TRACTCE'], right_on=['state','county','tract'])

    return used_tracts_df

if __name__ == '__main__':
    """
    Function to combined Airbnb dataset with
    TIGER/Line files to determine tracts in use.

    Function will then pull from the census API to
    load census data in combined file.

    Requires input of:
    TIGER/Line Shapefile (ESRI Shapefile)
    Airbnb Dataset (GeoJSON)
    Output location of resulting file (csv)
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_tiger_file', help='Input TIGER/Line file (SHP)')
    parser.add_argument(
        'input_airbnb_file', help='Input Airbnb file (GeoJSON)')
    parser.add_argument(
        'output_final_file', help='Output final file (csv)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )

    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        print('Combining files...')

    combined_df = combine_files(args.input_airbnb_file,args.input_tiger_file)

    if VERBOSE:
        print('Quering 2010 census...')
    df_2010 = query_census_2010(combined_df)

    if VERBOSE:
        print('Quering 2020 census...')
    df_2020 = query_census_2020(combined_df)

    if VERBOSE:
        print('Generating final file...')
    r_df = generate_final(combined_df,df_2010,df_2020)
    r_df.to_file(args.output_final_file)