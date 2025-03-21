# import os
import itertools
from functools import reduce

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from dbscan import DBSCAN

from configs.config_parser import Config
from configs.validators import validate_and_log
from .util import arc2psp, utc2my, is_sequence, download_index, print_stats

from Core.visualize import Mapper

CONF = Config('configs/config.yaml')

class RdrFilter():
    
    def __init__(self, 
                 url: str = CONF.URL, 
                 filepaths: list[str] = CONF.FILES, 
                 savedir: str = CONF.INDX_DIR):
        """
        Initialize RdrFilter instance.

        Args:
            url (str): The URL to download index files.
            filepaths (list[str]): List of filepaths to download.
            savedir (str): Directory where the index files are saved.
        """
        download_index(url, filepaths, savedir)
        self.local_conf = {}
        self.df = None
        self.tmp_df = None
        self.mapper = Mapper()
            
    def _assert_cluster(self):
        if 'CLUSTER' not in self.df.columns:
            raise Exception("Current filter can't be applied before cluster_filter()")

    def load_df(self,
                lbl_path: str = CONF.LBL_PATH,
                tab_path: str = CONF.TAB_PATH, 
                columns: list[str] = CONF.columns):
        
        """
        Load the dataframe from label and tab files. Keep only the specified columns.
        Discard the images that are not single-channel RED. Convert the areocentric
        co-ordinates to 2D polar stereographic.

        Args:
            lbl_path (str): Path to the label file.
            tab_path (str): Path to the tabular data file.
            columns (list[str]): List of columns to keep.
        """

        # load .tab
        with open(lbl_path, 'r') as f:
            labels = f.read().splitlines() 
        labels = [label.split()[2] for label in labels if 'NAME' in label]
        self.df = pd.read_csv(tab_path, names=labels)

        # keep only the useful columns; clean the PRODUCT_ID string; keep only the RED images (1 channel)
        self.df = self.df[columns]
        self.df['PRODUCT_ID'] = self.df['PRODUCT_ID'].str.strip()
        self.df['FILE_NAME_SPECIFICATION'] = self.df['FILE_NAME_SPECIFICATION'].str.strip()
        self.df = self.df[self.df['PRODUCT_ID'].str.contains('RED')]

        # convert the (LAT, LON) areocentric [ARC] co-oridantes to (X, Y) 2D polar stereographic [PSP]
        for i in range(1, 5):
            self.df.rename(columns={f'CORNER{i}_LONGITUDE':f'C{i}_LON', f'CORNER{i}_LATITUDE': f'C{i}_LAT'}, inplace=True)
            self.df[f'C{i}_X'], self.df[f'C{i}_Y'] = arc2psp(self.df[f'C{i}_LON'], self.df[f'C{i}_LAT'])
            
        # calculate image centroids
        for coor in ['LON', 'LAT', 'X', 'Y']:
            self.df[f'CTR_{coor}'] = self.df[[f'C{i}_{coor}' for i in range(1, 5)]].mean(axis=1)


        self.df.reset_index(drop=True, inplace=True)

        print_stats('NO FILTER', f'{len(self.df)} images')

    @validate_and_log
    def latitude_filter(self, 
                        min_lat: int = CONF.min_lat,
                        commit: bool = True):
        """
        Keep only the images above the specified minimum latitude. 
        Validate and log the parameters into the `local_conf` dictionary. 

        Args:
            min_lat (int): Minimum latitude value.
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            ValueError: If the minimum latitude is out of range.
        """
        self.tmp_df = self.df.copy()
        self.tmp_df = self.tmp_df[(self.tmp_df[[f'C{i}_LAT' for i in range(1,5)]] > min_lat).any(axis=1)]
        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('LATITUDE FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def scale_filter(self, 
                     scale: float = CONF.scale,
                     commit: bool = True):
        """
        Filter images by map scale.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args:
            scale (float): Desired map scale. Must be one of: 0.25, 0.5, 1.
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            ValueError: If an unsupported scale is provided.
        """
        self.tmp_df = self.df.copy()
        self.tmp_df = self.tmp_df[self.tmp_df['MAP_SCALE'] == scale]
        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('SCALE FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def season_filter(self, 
                      season: str = CONF.season,
                      commit: bool = True):
        """
        Filter images based on the specified season.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args:
            season (str): Season string to filter images by. 
                          Must be of `<hemisphere> <season>` format"
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            ValueError: If an unsupported season is provided.
        """

        if season in ('Northern spring', 'Southern autumn'):
            min_sol_long = 0
            max_sol_long = 90
        elif season in ('Northern summer', 'Southern winter'):
            min_sol_long = 90
            max_sol_long = 180
        elif season in ('Northern autumn', 'Southern spring'):
            min_sol_long = 180
            max_sol_long = 270
        elif season in ('Northern winter', 'Southern summer'):
            min_sol_long = 270
            max_sol_long = 360

        self.tmp_df = self.df.copy()
        self.tmp_df = self.tmp_df[(self.tmp_df['SOLAR_LONGITUDE'] > min_sol_long) & 
                        (self.tmp_df['SOLAR_LONGITUDE'] < max_sol_long)]
        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('SEASON FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def cluster_filter(self, 
                       algorithm: str = CONF.algorithm, 
                       min_samples: int = CONF.min_samples, 
                       epsilon: int = CONF.epsilon,
                       commit: bool = True):
        """
        Perform clustering on image centroids using the specified algorithm.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args:
            algorithm (str): Clustering algorithm to use. One of: `hdbscan` or `dbscan`.
            min_samples (int): Minimum numbers of points that can form a cluster.
            epsilon (int): Epsilon value (neighborhood size) for DBSCAN / HDBSCAN.
                           This should be provided in the distance units of the input data.
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            ValueError: If an unsupported algorithm is provided.
            ValueError: If min_samples < 2.
        """
        self.tmp_df = self.df.copy()

        # choose and algorithm and perform density based clustering of image centroids
        if algorithm == 'hdbscan':
            try:
                from hdbscan import HDBSCAN
                clusterer = HDBSCAN(min_cluster_size=2, 
                                    cluster_selection_epsilon = epsilon,
                                    min_samples=min_samples, 
                                    cluster_selection_method = 'leaf')
                clusterer.fit(self.tmp_df[['CTR_X', 'CTR_Y']].values)
                labels = clusterer.labels_
            except ModuleNotFoundError:
                print('''Module not pre-installed. Before calling again, run: 
                      $ conda install conda-forge::hdbscan''')
        elif algorithm == 'dbscan':
            labels, _ = DBSCAN(self.tmp_df[['CTR_X', 'CTR_Y']].values, 
                               min_samples = min_samples,
                               eps = epsilon)

        # create a new CLUSTER column & discard outlier images
        self.tmp_df['CLUSTER']  = labels
        self.tmp_df = self.tmp_df[self.tmp_df['CLUSTER'] != -1]
        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('CLUSTER FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def my_filter(self, 
                  min_years: int = CONF.min_years, 
                  mys: list[int] = CONF.mys, 
                  consecutive: bool = CONF.consecutive,
                  commit: bool = True):
        """
        Filter clusters based on the sequence of unique Mars Year (MY) it contains.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args:
            min_years (int): Minimum number of unique Mars Years to be present in a cluster.
            mys (list[int]): Specific Mars Years that must be present.
            consecutive (bool): Whether the years must be consecutive.
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            ValueError: If min_years < 2.
            Exception: If the current filter is applied before cluster_filter().
        """
        self._assert_cluster()

        # convert UTC time to Mars Year (MY). current range of MY is 27-35*
        self.tmp_df = self.df.copy()
        self.tmp_df['MY'] = self.tmp_df['OBSERVATION_START_TIME'].apply(utc2my)
        self.tmp_df.drop(['OBSERVATION_START_TIME'], axis = 1, inplace = True)

        valid_clusters = {}
        cluster_summary = self.tmp_df.groupby('CLUSTER')['MY'].unique()

        for cluster, years in cluster_summary.items():
            if mys and set(mys).issubset(years):
                valid_clusters[cluster] = years
            else:
                if len(years) >= min_years:
                    if consecutive:
                        if filtered_years := is_sequence(years, min_years):
                            valid_clusters[cluster] = filtered_years
                    else:
                        valid_clusters[cluster] = years
                    
        self.tmp_df = self.tmp_df[self.tmp_df.apply(lambda row: row['MY'] in valid_clusters.get(row['CLUSTER'], []), axis=1)]
        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('MY FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def keyword_filter(self, 
                       keywords: list[str] = CONF.keywords,
                       commit: bool = True):
        """
        Filter clusters based on the presence of specific keywords in the RATIONALE_DESC column.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args:
            keywords (list[str]): List of keywords to filter by.
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            Exception: If the current filter is applied before cluster_filter().
        """
        self._assert_cluster()

        self.tmp_df = self.df.copy()

        pattern = r'(?i)(?:' + '|'.join(keywords)  + r')'
        self.tmp_df = self.tmp_df.groupby("CLUSTER").filter(
            lambda group: group["RATIONALE_DESC"].str.contains(pattern).any()
        )

        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('KEYWORD FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def allignment_filter(self, 
                          commit: bool = True):
        """
        Filter and stack images based on the maximum overlapping area of their footprints.
        The function computes the intersection area of polygons defined by the image corners 
        and selects the best stack of images per cluster. Validate and log the parameters 
        into the `local_conf` dictionary. 

        Args:
            commit (bool): Whether to commit the changes to the main dataframe.
        Raises:
            Exception: If the current filter is applied before cluster_filter().
        """
        self._assert_cluster()
       
        self.tmp_df = self.df.copy()

        filtered_ids = []
        for cluster in self.tmp_df['CLUSTER'].unique():
            cluster_df = self.tmp_df[self.tmp_df['CLUSTER'] == cluster]

            group_indices = [
                cluster_df[cluster_df['MY'] == my].index.tolist()
                for my in cluster_df['MY'].unique()
            ]
            
            max_area = 0
            best_stack = None
            
            for stack in itertools.product(*group_indices):
               
                polygons = [
                    Polygon([
                        (self.tmp_df.loc[idx][f'C{i}_X'], self.tmp_df.loc[idx][f'C{i}_Y'])
                        for i in range(1,5)
                    ]) 
                    for idx in stack
                ]
                
                intersection = reduce(lambda p1, p2: p1.intersection(p2), polygons)
                if intersection.area > max_area:
                    max_area = intersection.area
                    best_stack = stack
            
            if best_stack is not None:
                filtered_ids.extend(best_stack)
        
        self.tmp_df = self.tmp_df.loc[filtered_ids]

        self.tmp_df.reset_index(drop=True, inplace=True)

        print_stats('ALLIGNMENT FILTER', f'{len(self.tmp_df)} images') 

        if commit:
            self.df = self.tmp_df.copy()

    @validate_and_log
    def visualize(self, 
                  target: str, 
                  engine: str, 
                  filename: str):
        """
        Visualize the filtered images using the specified engine.
        Only validate the parameters without loggin them.

        Args:
            target (str): Target to visualize. One of: `img_rectangle`, `img_centroid`, `cluster_centroid`.
            engine (str): Engine to use for visualization. One of: `pygmt`, `qgis`.
            filename (str): Name of the output file.
        Raises:
            ValueError: If an unsupported target is provided.
            ValueError: If an unsupported engine is provided.
            Exception: If the current filter is applied before cluster_filter().
        """
        
        if target == 'cluster_centroid':
            if 'CLUSTER' in df.columns:
                df = df.groupby('CLUSTER')[['CTR_LON', 'CTR_LAT', 'CTR_X', 'CTR_Y']].mean().reset_index()
            else:
                raise Exception("Cluster centroids can't be mapped before cluster_filter() is appplied")
           
        if engine == 'qgis':
            self.mapper.use_qgis(self.tmp_df, target, filename)
        elif engine == 'pygmt':
            self.mapper.use_pygmt(self.tmp_df, target, filename)

    def save_df(self, filepath: str = CONF.FILTERED_PATH):
        self.df.to_csv(filepath, sep="\t", index=False)


if __name__ == '__main__':
    explorer = RdrFilter()

    if CONF.filter_sequence:
        for filter in CONF.filter_sequence:
            getattr(explorer, filter)()
    else:
        raise ValueError('Define a non-empty filter_sequence in `configs.yaml`.')