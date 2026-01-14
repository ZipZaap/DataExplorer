import os
import pygmt 
import tempfile
import pandas as pd
import geojson as gj
from typing import Literal
from pathlib import Path

from configs.config_parser import Config

CONF = Config('configs/config.yaml')

class MapPlotter:
    """
    Class for generating and displaying maps using PyGMT.
    """
    def __init__(self,
                 resolution: Literal["30m", "20m", "15m", "10m"] = CONF.resolution,
                 map_region: str = CONF.map_region,
                 projection: str = CONF.projection,
                 pygmt_savedir: Path = CONF.MAP_DIR,
                 qgis_savedir: Path = CONF.QGIS_DIR):
        """
        Initialize the PyGMT instance with given map parameters.
        Loads the relief grid for Mars from PyGMT datasets.

        Args
        ----
            resolution : str
                Resolution for the Mars relief grid.

            map_region: str
                Geographic region to display on the map.

            projection : str
                Map projection to use.

            pygmt_savedir : Path
                Directory to save the generated PyGMT image.

            qgis_savedir : Path
                Directory to save the generated GeoJSON file.
        """

        self.qgis_savedir = qgis_savedir
        self.pygmt_savedir = pygmt_savedir

        self.map_region = map_region
        self.projection = projection
        self.mars_grid = pygmt.datasets.load_mars_relief(resolution=resolution, region=map_region)
    
    def visualize(self,
                  df: pd.DataFrame,
                  target: str, 
                  color: str = "purple",
                  engine: str = "pygmt", 
                  title: str | None = None):
        """
        Visualize the filtered images using the specified engine. 
        If title is provided, save the output to a file. Otherwise, display it interactively.

        Args
        ----
            df : pd.DataFrame
                DataFrame containing the filtered images to visualize.

            target : str
                Target to visualize. One of: `img_footprint`, `img_centroid`, `cluster`.

            color : str
                Target color.

            engine : str
                Engine to use for visualization. One of: `pygmt`, `qgis`.

            title : str | None
                Name of the output file.

        Raises
        ------
            ValueError
                If an unsupported target is provided.

            ValueError
                If an unsupported engine is provided.

            RuntimeError
                If visualization of cluster centroids is attempted before cluster_filter().
            
            RuntimeError
                If visualization is attempted before any filter is applied.
        """

        if df is None or df.empty:
            raise RuntimeError("Dataframe is empty, please apply some filter before visualization")

        if target not in {'img_footprint', 'img_centroid', 'cluster'}:
            raise ValueError("Unsupported target! Choose one of: `img_footpirnt`, `img_centroid`, `cluster`.")
        elif target == 'cluster':
            if 'CLUSTER' not in df.columns:
                raise RuntimeError("Cluster centroids can't be mapped before cluster_filter() is applied")
            df = df.groupby('CLUSTER')[['CTR_LON', 'CTR_LAT', 'CTR_X', 'CTR_Y']].mean().reset_index()

        if engine == 'qgis':
            if title:
                self._use_qgis(df, target, title)
            else:
                raise ValueError("Please provide a title to save the GeoJSON file.")
        elif engine == 'pygmt':
            self._use_pygmt(df, target, color, title)
        else:
            raise ValueError("Unsupported engine! Choose one of: `pygmt` or `qgis`.")

    def _use_pygmt(self,
                   df: pd.DataFrame,
                   target: str,
                   color: str,
                   title: str | None):
        """
        Display a map with overlays based on the target and data provided.
        Optionally save the map to a file.

        Args
        ----
            df : pd.DataFrame
                DataFrame containing coordinate data.

            target : str
                Type of overlay. One of 'img_footprint', 'img_centroid' or 'cluster'.

            title : str | None
                Title of the output file (without extension).
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
            if target in ['img_centroid', 'cluster']:
                for _, row in df.iterrows():
                    tmp.write(">\n")
                    tmp.write(f"{row['CTR_LON']} {row['CTR_LAT']}\n")
                tmp.flush()
                style = "c0.3"
            else: # target == 'img_footprint':
                for _, row in df.iterrows():
                    tmp.write(">\n")
                    tmp.write(f"{row['C1_LON']} {row['C1_LAT']}\n")
                    tmp.write(f"{row['C2_LON']} {row['C2_LAT']}\n")
                    tmp.write(f"{row['C3_LON']} {row['C3_LAT']}\n")
                    tmp.write(f"{row['C4_LON']} {row['C4_LAT']}\n")
                tmp.flush()
                style = None
           
            
        fig.plot(data=tmp.name, fill=color, pen="0.2p,black", style=style, transparency=20)

        if title:
            os.makedirs(self.pygmt_savedir, exist_ok=True)
            fig.savefig(os.path.join(self.pygmt_savedir, f"{title}.png"))
        else:
            fig.show(width=700)

        os.remove(tmp.name)

    def _use_qgis(self,
                  df: pd.DataFrame, 
                  target: str,
                  title: str):
        """
        Save a QGIS layer as a GeoJSON file based on the target type.

        Args:
            df : pd.DataFrame
                DataFrame containing coordinate and attribute data.

            target : str
                Type of layer to generate. One of: `img_footprint`, `img_centroid` or `cluster`.
                
            title : str
                Title of the output file (without extension).
        """
        
        features = []
        if target == 'img_footprint':
            for idx, row in df.iterrows(): 
                polygon = gj.Polygon([[(row[f"C{i}_X"], row[f"C{i}_Y"]) for i in [1,2,3,4,1]]])
                features.append(gj.Feature(geometry=polygon, properties={"ID": idx, "Name": row.PRODUCT_ID}))
        elif target == 'img_centroid':
            for idx, row in df.iterrows(): 
                point = gj.Point((row['CTR_X'], row['CTR_Y']))
                features.append(gj.Feature(geometry=point, properties={"ID": idx, "Name": row.PRODUCT_ID}))
        elif target == 'cluster':
            for idx, row in df.iterrows(): 
                point = gj.Point((row['CTR_X'], row['CTR_Y']))
                features.append(gj.Feature(geometry=point, properties={"CLUSTER": row.CLUSTER}))
            
        feature_collection = gj.FeatureCollection(features)

        os.makedirs(self.qgis_savedir, exist_ok=True)
        with open(os.path.join(self.qgis_savedir , f"{title}.geojson"), 'w') as f:
            gj.dump(feature_collection, f)
