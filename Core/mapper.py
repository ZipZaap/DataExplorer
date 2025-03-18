import os
import tempfile
import pandas as pd

import pygmt 
import geojson as gj

from configs.config_parser import Config
CONF = Config('configs/config.yaml')

class PyGMT:
    """
    Class for generating and displaying maps using PyGMT.
    """
    def __init__(self,
                 resolution: str = CONF.resolution,
                 min_lat: int = CONF.min_lat,
                 savedir: str = CONF.MAPS_DIR):
        """
        Initialize the PyGMT instance with given map parameters.
        Loads the relief grid for Mars from PyGMT datasets.

        Args:
            resolution (str): Resolution for the Mars relief grid.
            min_lat (int): Minimum latitude for the map region.
            savedir (str): Directory to save the generated map image.
        """
        self.savedir= savedir
        self.region = f"0/360/{min_lat}/90"
        self.mars_grid = pygmt.datasets.load_mars_relief(resolution = resolution, region = self.region)
       
    def show_on_map(self,
                    df: pd.DataFrame,
                    target: str,
                    filename: str = None):
        """
        Display a map with overlays based on the target and data provided.
        Optionally save the map to a file.

        Args:
            df (pd.DataFrame): DataFrame containing coordinate data.
            target (str): Type of overlay. Must be one of 'img_rectangle', 'img_centroid',
                          or 'cluster_centroid'.
            filename (str, optional): If provided, saves the map as a PNG file in savedir.

        Raises:
            Exception: If target is 'cluster_centroid' but the 'CLUSTER' column is missing.
            ValueError: If an invalid target is provided.
        """
        
        if target == 'cluster_centroid':
            if 'CLUSTER' in df.columns:
                df = df.groupby('CLUSTER')[['CTR_LON', 'CTR_LAT', 'CTR_X', 'CTR_Y']].mean().reset_index()
            else:
                raise Exception("Cluster centroids can't be mapped before cluster_filter() is appplied")
        
        fig = pygmt.Figure()
        pygmt.makecpt(cmap='dem3', series=[-8000, -1800, 1], continuous=True)

        fig.grdimage(
            grid=self.mars_grid, 
            region=self.region,
            projection="G0/90/12c", 
            frame=["xa30"], 
            cmap=False, 
            dpi=250
        )

        fig.grdcontour(
            grid=self.mars_grid, 
            annotation=None, 
            levels=250, 
            limit=[-4800, -2900], 
            pen="0.1p", 
            transparency = 30
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            if target == 'img_rectangle':
                for _, row in df.iterrows():
                    tmp.write(">\n")
                    tmp.write(f"{row['C1_LON']} {row['C1_LAT']}\n")
                    tmp.write(f"{row['C2_LON']} {row['C2_LAT']}\n")
                    tmp.write(f"{row['C3_LON']} {row['C3_LAT']}\n")
                    tmp.write(f"{row['C4_LON']} {row['C4_LAT']}\n")
                tmp.flush()
                style = None
            elif target in ['img_centroid', 'cluster_centroid']:
                for _, row in df.iterrows():
                    tmp.write(">\n")
                    tmp.write(f"{row['CTR_LON']} {row['CTR_LAT']}\n")
                tmp.flush()
                style = "c0.1"
            else:
                raise ValueError("Invalid target! Choose one of: `img_rectangle`, `img_centroid`, `cluster_centroid`.")
            
        fig.plot(data=tmp.name, fill="cyan", pen="0.2p,black", style=style, transparency=20)
        fig.show(width=700)
        
        if filename:
            save_path = os.path.join(self.savedir, f"{filename}.png")
            fig.savefig(save_path)

        os.remove(tmp.name)

def save_qgis_layer(df: pd.DataFrame, 
                    target: str,
                    filename: str,
                    savedir: str = CONF.QGIS_DIR
                    ) -> None:
    """
    Save a QGIS layer as a GeoJSON file based on the target type.

    Args:
        df (pd.DataFrame): DataFrame containing coordinate and attribute data.
        target (str): Type of layer to generate. Must be one of `img_rectangle`, `img_centroid` or `cluster_centroid`.
        filename (str): Name of the output file (without extension).
        savedir (str, optional): Directory where the file will be saved.

    Raises:
        Exception: If target is `cluster_centroid` but the `CLUSTER` column is missing.
        ValueError: If an invalid target is provided.
    """
        
    if target == 'cluster_centroid':
        if 'CLUSTER' in df.columns:
            df = df.groupby('CLUSTER')[['CTR_LON', 'CTR_LAT', 'CTR_X', 'CTR_Y']].mean().reset_index()
        else:
            raise Exception("Cluster centroids can't be mapped before cluster_filter() is appplied")

    features = []
    if target == 'img_rectangle':
        for idx, row in df.iterrows(): 
            polygon = gj.Polygon([(row[f"C{i}_X"], row[f"C{i}_Y"]) for i in [1,2,3,4,1]])
            features.append(gj.Feature(geometry=polygon, properties={"ID": idx, "Name": row.PRODUCT_ID}))
    elif target == 'img_centroid':
        for idx, row in df.iterrows(): 
            point = gj.Point((row['CTR_X'], row['CTR_Y']))
            features.append(gj.Feature(geometry=point, properties={"ID": idx, "Name": row.PRODUCT_ID}))
    elif target == 'cluster_centroid':
        for idx, row in df.iterrows(): 
            point = gj.Point((row['CTR_X'], row['CTR_Y']))
            features.append(gj.Feature(geometry=point, properties={"CLUSTER": row.CLUSTER}))
    else:
        raise ValueError("Invalid target! Choose one of: `img_rectangle`, `img_centroid` or `cluster_centroid`.")
        
    feature_collection = gj.FeatureCollection(features)

    save_path = os.path.join(savedir, f"{filename}.geojson")
    with open(save_path, 'w') as f:
        gj.dump(feature_collection, f)
