import os
import tempfile
import pandas as pd

import pygmt 
import geojson as gj

from configs.config_parser import Config
CONF = Config('configs/config.yaml')

class Mapper:
    """
    Class for generating and displaying maps using PyGMT.
    """
    def __init__(self,
                 resolution: str = CONF.resolution,
                 map_region: str = CONF.map_region,
                 projection: str = CONF.projection,
                 pygmt_savedir: str = CONF.MAPS_DIR,
                 qgis_savedir: str = CONF.QGIS_DIR
                 ):
        """
        Initialize the PyGMT instance with given map parameters.
        Loads the relief grid for Mars from PyGMT datasets.

        Args:
            resolution (str): Resolution for the Mars relief grid.
            map_region (str): Geographic region to display on the map.
            projection (str): Map projection to use.
            pygmt_savedir (str): Directory to save the generated PyGMT image.
            qgis_savedir (str): Directory to save the generated GeoJSON file.
        """
        self.qgis_savedir = qgis_savedir
        self.pygmt_savedir = pygmt_savedir

        self.map_region = map_region
        self.projection = projection
        self.mars_grid = pygmt.datasets.load_mars_relief(resolution = resolution, region = map_region)
       
    def use_pygmt(self,
                  df: pd.DataFrame,
                  target: str,
                  filename: str
                 ) -> None:
        """
        Display a map with overlays based on the target and data provided.
        Optionally save the map to a file.

        Args:
            df (pd.DataFrame): DataFrame containing coordinate data.
            target (str): Type of overlay. Must be one of 'img_rectangle', 'img_centroid',
                          or 'cluster_centroid'.
            filename (str): Name of the output file (without extension).
        """
        
        fig = pygmt.Figure()
        pygmt.makecpt(cmap='dem3', series=[-8000, -1800, 1], continuous=True)

        fig.grdimage(
            grid=self.mars_grid, 
            region=self.map_region,
            projection=self.projection, 
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
                style = "c0.3"
            
        fig.plot(data=tmp.name, fill="purple", pen="0.2p,black", style=style, transparency=20)
        # fig.show(width=700)
        fig.savefig(os.path.join(self.pygmt_savedir, f"{filename}.png"))

        os.remove(tmp.name)

    def use_qgis(self,
                 df: pd.DataFrame, 
                 target: str,
                 filename: str
                ) -> None:
        """
        Save a QGIS layer as a GeoJSON file based on the target type.

        Args:
            df (pd.DataFrame): DataFrame containing coordinate and attribute data.
            target (str): Type of layer to generate. Must be one of `img_rectangle`, `img_centroid` or `cluster_centroid`.
            filename (str): Name of the output file (without extension).
        """

        features = []
        if target == 'img_rectangle':
            for idx, row in df.iterrows(): 
                polygon = gj.Polygon([[(row[f"C{i}_X"], row[f"C{i}_Y"]) for i in [1,2,3,4,1]]])
                features.append(gj.Feature(geometry=polygon, properties={"ID": idx, "Name": row.PRODUCT_ID}))
        elif target == 'img_centroid':
            for idx, row in df.iterrows(): 
                point = gj.Point((row['CTR_X'], row['CTR_Y']))
                features.append(gj.Feature(geometry=point, properties={"ID": idx, "Name": row.PRODUCT_ID}))
        elif target == 'cluster_centroid':
            for idx, row in df.iterrows(): 
                point = gj.Point((row['CTR_X'], row['CTR_Y']))
                features.append(gj.Feature(geometry=point, properties={"CLUSTER": row.CLUSTER}))
            
        feature_collection = gj.FeatureCollection(features)

        with open(os.path.join(self.qgis_savedir , f"{filename}.geojson"), 'w') as f:
            gj.dump(feature_collection, f)
