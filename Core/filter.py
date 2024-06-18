import numpy as np
import pandas as pd
from datetime import datetime

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
import shapely.wkt


from .util import planetocentric2map, qgis_geojson_layer, line_intersection, cluster_centroids, get_centroid
from configs import CONF

def loadtab(tab_path = CONF.TAB_PATH, lbl_path = CONF.LBL_PATH, min_lat = CONF.MIN_LAT, columns = CONF.COLUMNS):
    # load the .lbl file and read the database csv
    with open(lbl_path, 'r') as f:
        labels = f.read().splitlines() 
    labels = [label.split()[2] for label in labels if 'NAME' in label]
    df = pd.read_csv(tab_path, names=labels)
    df.reset_index(drop=True, inplace = True)

    # clean the ID string from trailing.leading spaces and only keep the RED images (1 channel)
    df.loc[:,'PRODUCT_ID'] = df.PRODUCT_ID.apply(lambda x: x.strip())
    df = df[df.PRODUCT_ID.apply(lambda x: 'RED' in x.split('_')[-1])]

    # select the columns of interest
    df = df[columns]

    # filter out images below 78 deg latitude, i.e. keeping only the northern himisphere in the NPLD vicinity
    df = df[(df.CORNER1_LATITUDE > min_lat) & (df.CORNER2_LATITUDE > min_lat) & \
            (df.CORNER3_LATITUDE > min_lat) & (df.CORNER4_LATITUDE > min_lat)]
    
    for i in range(1,5):
        df.loc[:,f'C{i}'] = df.apply(lambda x: planetocentric2map(x[f'CORNER{i}_LATITUDE'], x[f'CORNER{i}_LONGITUDE']), axis = 1)
        df.drop([f'CORNER{i}_LATITUDE', f'CORNER{i}_LONGITUDE'], axis = 1, inplace = True)

    # calculate the center and the area of each image on a 2D map
    df['CENTER'] = ''
    df['CENTER'] = df.apply(lambda img: line_intersection((img.C1, img.C3), (img.C2, img.C4)), axis = 1)
    df['AREA'] = ''
    df['AREA'] = df.apply(lambda img: Polygon([img.C1, img.C2, img.C3, img.C4]).area, axis = 1)

    # convert UTC time to Mars Year (MY). current range of MY is 27-35*
    df.loc[:,'MY'] = df.OBSERVATION_START_TIME.apply(lambda x: int(abs((datetime.strptime('1955-4-11T00:00:00', "%Y-%m-%dT%H:%M:%S") - datetime.strptime(x.strip(), '%Y-%m-%dT%H:%M:%S')).days)/687))
    df.reset_index(drop=True, inplace = True)

    return df

def scale_filter(df, scale = CONF.SCALE, sv_layer = False):
    # filter out images with map scale greater than 0.25 m/px
    df = df[df.MAP_SCALE == scale]
    if sv_layer:
        qgis_geojson_layer(df, 'scale_filter')
    return df

def keyword_filter(df, keywords = CONF.KEYWORDS , sv_layer = False):
    # keep only images containig the keywords in the rationale description
    df = df[df['RATIONALE_DESC'].str.contains('|'.join(keywords))]
    if sv_layer:
        qgis_geojson_layer(df, 'keyword_filter')
    return df

def season_filter(df, season = CONF.SEASON, sv_layer = False):
    if season == 'Northern spring' or season == 'Southern autumn':
        min_sol_long = 0
        max_sol_long = 90
    elif season == 'Northern summer' or season == 'Southern winter':
        min_sol_long = 90
        max_sol_long = 180
    elif season == 'Northern autumn' or season == 'Southern spring':
        min_sol_long = 180
        max_sol_long = 270
    elif season == 'Northern winter' or season == 'Southern summer':
        min_sol_long = 270
        max_sol_long = 360
    df = df[(df.SOLAR_LONGITUDE > min_sol_long) & (df.SOLAR_LONGITUDE < max_sol_long)]
    if sv_layer:
        qgis_geojson_layer(df, 'season_filter')
    return df
        
def cluster_filter(df, sv_layer = False):
    df = cluster_centroids(df)
    df = df[df['CLUSTER'] != -1]

    centroids_df = df.groupby(['CLUSTER'], as_index = False ).agg({'CENTER': lambda x: get_centroid(x)})
    if sv_layer:
        qgis_geojson_layer(df, 'clustered_imgs', geom = 'poly')
        qgis_geojson_layer(centroids_df, 'cluster_centroids', geom = 'point')
    return df
    
# def get_largest_image(df):
#     df = cluster_centroids(df)
#     df = df.loc[df.groupby('CLUSTER')['AREA'].idxmax()].reset_index(drop = True)
#     qgis_geojson_layer(df, 'largest_img_per_cluster')


