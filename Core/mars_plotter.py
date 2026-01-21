import os
import pygmt 
import tempfile
import pandas as pd
import geojson as gj
from typing import Literal
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from configs.config_parser import Config

CONF = Config('configs/config.yaml')

class MapPlotter():
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
                style = "c0.25"
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

        # --- NEW CODE BLOCK START ---
        # If the target is 'cluster', add text labels next to the points
        if target == 'cluster':
            fig.text(
                x=df['CTR_LON'].values,
                y=df['CTR_LAT'].values,
                text=df['CLUSTER'].astype(str).values,
                font="7p,Helvetica-Bold,black",       
                offset="0.1c/-0.1c"
            )
        # --- NEW CODE BLOCK END ---

        if title:
            # os.makedirs(self.pygmt_savedir, exist_ok=True)
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

        # os.makedirs(self.qgis_savedir, exist_ok=True)
        with open(os.path.join(self.qgis_savedir , f"{title}.geojson"), 'w') as f:
            gj.dump(feature_collection, f)

    def show_on_map(self,
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
                If visualization with `qgis` is attempted without setting the `title`.

            ValueError
                If an unsupported engine is provided.
        """

        if engine == 'qgis':
            if title:
                self._use_qgis(df, target, title)
            else:
                raise ValueError("Please provide a title to save the GeoJSON file.")
        elif engine == 'pygmt':
            self._use_pygmt(df, target, color, title)
        else:
            raise ValueError("Unsupported engine! Choose one of: `pygmt` or `qgis`.")
        
    def show_preview(self, 
                     df: pd.DataFrame, 
                     imdir: Path):
        """
        Display a grid of image previews organized by Mars Year.

        This method creates a matplotlib figure where each column corresponds to a 
        unique Mars Year (MY) present in the dataframe, and rows contain the 
        thumbnail images associated with that year.

        Args
        ----
            df : pd.DataFrame
                DataFrame containing the image metadata. 
                Must include 'MY' and 'PRODUCT_ID' columns.

            imdir : Path
                Directory containing the downloaded thumbnail images (.jpg).
        """

        # Get unique bins (columns) and max products per bin (rows)
        unique_my = df['MY'].unique()
        n_cols = len(unique_my)
        n_rows = df.groupby('MY').size().max()

        # Create figure with dynamic size based on grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

        # Turn off axes for all subplots initially (clean slate)
        for ax in axes.flat:
            ax.axis('off')

        for col_idx, my in enumerate(unique_my):
            bin_data = df[df['MY'] == my].reset_index(drop=True)
            
            # We align x to the axes (center of column) and y to the figure (top of page)
            top_ax = axes[0, col_idx]
            trans = transforms.blended_transform_factory(top_ax.transAxes, fig.transFigure)
            
            # Place label at y=0.93 (near top of figure), x=0.5 (center of axes)
            top_ax.text(0.5, 0.97, f"MY {my}", transform=trans, 
                        ha='center', va='top', fontsize=14, fontweight='bold', color='black')

            # Loop through each Product within the Mars Year
            for row_idx in range(n_rows):
                ax = axes[row_idx, col_idx]

                if row_idx < len(bin_data):
                    prod_id = bin_data.loc[row_idx, 'PRODUCT_ID']
                    impath = imdir / f'{prod_id}.thumb.jpg'

                    try:
                        with Image.open(impath) as img_obj:
                            img_gray = img_obj.convert('L')
                            ax.imshow(img_gray, cmap='gray')

                        # --- LABEL PRODUCT ID (Bottom Left, Pink, Bold) ---
                        bbox_props = dict(facecolor='black', edgecolor='none', pad=0)
                        ax.text(0.00, 0.00, str(prod_id), transform=ax.transAxes,
                                color='#FF00FF', fontweight='bold', fontsize=10,
                                ha='left', va='bottom', bbox=bbox_props)

                    except FileNotFoundError:
                        ax.text(0.5, 0.5, "Image\nNot Found", ha='center', va='center', transform=ax.transAxes)
                        for spine in ax.spines.values():
                            spine.set_visible(True)

            plt.tight_layout()