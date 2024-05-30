import shapefile as shp
import geojson as gj
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

from configs import CONF

def qgis_shp_layer(data, fname, geom = 'poly'):

    fpath = f'{CONF.RDR_DIR}/{fname}.shp'
    # Create a shapefile writer
    with shp.Writer(fpath) as f:
        if geom == 'poly':
            f.shapeType = shp.POLYGON
            
            # Define the fields
            f.field('ID', 'N')  # Numeric field for polygon ID
            f.field('Name', 'C')  # Character field for polygon name
            
            # Add polygons with unique IDs
            for idx, val in data.iterrows():
                f.record(ID = idx, Name = val.PRODUCT_ID)
                f.poly([[val.C1, val.C2, val.C3, val.C4, val.C1]])

        elif geom == 'point':
            f.shapeType = shp.POINT
            
            # Define the fields
            f.field('CLUSTER', 'N')

            # Add points with unique CLUSTER IDs
            for idx, val in data.iterrows():
                f.record(ID = val.CLUSTER)
                f.point(val.CENTER[0], val.CENTER[1])

def qgis_geojson_layer(data, fname, geom = 'poly'):
    fpath = f'{CONF.RDR_DIR}/{fname}.geojson'

    features = []
    for idx, val in data.iterrows():
        if geom == 'poly':
            polygon = gj.Polygon([[val.C1, val.C2, val.C3, val.C4, val.C1]])
            features.append(gj.Feature(geometry=polygon, properties={"ID": idx, "Name": val.PRODUCT_ID}))
        elif geom == 'point':
            point = gj.Point(val.CENTER)
            features.append(gj.Feature(geometry=point, properties={"CLUSTER": val.CLUSTER}))

    # Create a GeoJSON FeatureCollection
    feature_collection = gj.FeatureCollection(features)
    # Write the GeoJSON file
    with open(fpath, 'w') as f:
        gj.dump(feature_collection, f)


def planetocentric2map(LAT, LON):
    x = round(2*3376.2*np.tan(np.pi/4 - np.radians(LAT/2))*np.sin(np.radians(LON)) * 1000, 2)
    y = round(-2*3376.2*np.tan(np.pi/4 - np.radians(LAT/2))*np.cos(np.radians(LON)) * 1000, 2)
    return (x, y)

# calculate the center of the image on the 2D map
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = round(det(d, xdiff) / div, 2)
    y = round(det(d, ydiff) / div, 2)
    return (x, y)

def cluster_centroids(df, algotype = 'dbscan'):
    if algotype == 'hdbscan':
        clusterer = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_method = 'leaf', cluster_selection_epsilon = 2700)
        clusterer.fit(df.CENTER.tolist())
    elif algotype == 'dbscan':
        clusterer = DBSCAN(eps=2700, min_samples=2)
        clusterer.fit(df.CENTER.tolist())
    df.loc[:,'CLUSTER'] = clusterer.labels_

    return df

def get_centroid(x):
    x = np.round(np.mean(np.vstack(x), axis = 0), 2)
    return (x[0], x[1])

def extract_imIDs(fname, sv_txt = False):
    # Load the GeoJSON file
    fpath = f'{CONF.RDR_DIR}/{fname}.geojson'
    with open(fpath, 'r') as f:
        data = gj.load(f)

    imIDs = []
    for feature in data.features:
        imIDs.append(feature.properties['Name']) #.replace('_RED', '')

    if sv_txt:
        with open('imIDs.txt', 'w') as f:
            for item in imIDs:
                f.write("%s\n" % item)
    else:
        return imIDs
    