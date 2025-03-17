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
            
    def _assert_cluster(self):
        if 'CLUSTER' not in self.df.columns:
            raise Exception("Current filter can't be applied before cluster_filter()")

    def load_df(self,
                lbl_path: str = CONF.LBL_PATH,
                tab_path: str = CONF.TAB_PATH, 
                columns: list[str] = CONF.columns):
        
        """
        Load the dataframe from label and tab files.

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

        Args:
            min_lat (int): Minimum latitude value.
        """
        tmp_df = self.df.copy()
        tmp_df = tmp_df[(tmp_df[[f'C{i}_LAT' for i in range(1,5)]] > min_lat).any(axis=1)]
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('LATITUDE FILTER', f'{len(tmp_df)} images')

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df

    @validate_and_log
    def scale_filter(self, 
                     scale: float = CONF.scale,
                     commit: bool = True):
        """
        Filter images by map scale.

        Args:
            scale (float): Desired map scale.
        """
        tmp_df = self.df.copy()
        tmp_df = tmp_df[tmp_df['MAP_SCALE'] == scale]
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('SCALE FILTER', f'{len(tmp_df)} images')

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df

    @validate_and_log
    def season_filter(self, 
                      season: str = CONF.season,
                      commit: bool = True):
        """
        Filter images based on the specified season.

        Args:
            season (str): Season string to filter images by.
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

        tmp_df = self.df.copy()
        tmp_df = tmp_df[(tmp_df['SOLAR_LONGITUDE'] > min_sol_long) & 
                        (tmp_df['SOLAR_LONGITUDE'] < max_sol_long)]
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('SEASON FILTER', f'{len(tmp_df)} images')

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df

    @validate_and_log
    def cluster_filter(self, 
                       algorithm: str = CONF.algorithm, 
                       min_samples: int = CONF.min_samples, 
                       epsilon: int = CONF.epsilon,
                       commit: bool = True):
        """
        Perform clustering on image centroids using the specified algorithm.

        Args:
            algorithm (str): Clustering algorithm to use ('hdbscan' or 'dbscan').
            min_samples (int): Minimum samples required for clustering.
            epsilon (int): Epsilon value (neighborhood size) for DBSCAN / HDBSCAN.
                           This should be provided in the distance units of the input data.
        """
        tmp_df = self.df.copy()

        # choose and algorithm and perform density based clustering of image centroids
        if algorithm == 'hdbscan':
            try:
                from hdbscan import HDBSCAN
                clusterer = HDBSCAN(min_cluster_size=2, 
                                    cluster_selection_epsilon = epsilon,
                                    min_samples=min_samples, 
                                    cluster_selection_method = 'leaf')
                clusterer.fit(tmp_df[['CTR_X', 'CTR_Y']].values)
                labels = clusterer.labels_
            except ModuleNotFoundError:
                print('''Module not pre-installed. Before calling again, run: 
                      $ conda install conda-forge::hdbscan''')
        elif algorithm == 'dbscan':
            labels, _ = DBSCAN(tmp_df[['CTR_X', 'CTR_Y']].values, 
                               min_samples = min_samples,
                               eps = epsilon)

        # create a new CLUSTER column & discard outlier images
        tmp_df['CLUSTER']  = labels
        tmp_df = tmp_df[tmp_df['CLUSTER'] != -1]
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('CLUSTER FILTER', f'{len(tmp_df)} images')

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df

    @validate_and_log
    def my_filter(self, 
                  min_years: int = CONF.min_years, 
                  mys: list[int] = CONF.mys, 
                  consecutive: bool = CONF.consecutive,
                  commit: bool = True):
        """
        Filter clusters based on Mars Year (MY) criteria.

        Args:
            min_years (int): Minimum number of Mars Years to be present in a cluster.
            mys (list[int]): Specific Mars Years that must be present.
            consecutive (bool): Whether the years must be consecutive.
        """
        self._assert_cluster()

        # convert UTC time to Mars Year (MY). current range of MY is 27-35*
        tmp_df = self.df.copy()
        tmp_df['MY'] = tmp_df['OBSERVATION_START_TIME'].apply(utc2my)
        tmp_df.drop(['OBSERVATION_START_TIME'], axis = 1, inplace = True)

        valid_clusters = {}
        cluster_summary = tmp_df.groupby('CLUSTER')['MY'].unique()

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
                    
        tmp_df = tmp_df[tmp_df.apply(lambda row: row['MY'] in valid_clusters.get(row['CLUSTER'], []), axis=1)]
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('MY FILTER', f'{len(tmp_df)} images')

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df

    @validate_and_log
    def keyword_filter(self, 
                       keywords: list[str] = CONF.keywords,
                       commit: bool = True):
        """
        Filter clusters based on the presence of specific keywords in the DESCRIPTION column.

        Args:
            keywords (list[str]): List of keywords to filter by.
        """
        self._assert_cluster()

        tmp_df = self.df.copy()

        pattern = r'(?i)(?:' + '|'.join(keywords)  + r')'
        tmp_df = tmp_df.groupby("CLUSTER").filter(
            lambda group: group["RATIONALE_DESC"].str.contains(pattern).any()
        )

        # reset index
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('KEYWORD FILTER', f'{len(tmp_df)} images')

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df

    @validate_and_log
    def allignment_filter(self, 
                          commit: bool = True):
        """
        Filter and stack images based on the maximum overlapping area of their footprints.
        The function computes the intersection area of polygons defined by the image corners 
        and selects the best stack of images per cluster.
        """
        self._assert_cluster()
       
        tmp_df = self.df.copy()

        filtered_ids = []
        for cluster in tmp_df['CLUSTER'].unique():
            cluster_df = tmp_df[tmp_df['CLUSTER'] == cluster]

            group_indices = [
                cluster_df[cluster_df['MY'] == my].index.tolist()
                for my in cluster_df['MY'].unique()
            ]
            
            max_area = 0
            best_stack = None
            
            for stack in itertools.product(*group_indices):
               
                polygons = [
                    Polygon([
                        (tmp_df.loc[idx][f'C{i}_X'], tmp_df.loc[idx][f'C{i}_Y'])
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
        
        tmp_df = tmp_df.loc[filtered_ids]

        # reset index
        tmp_df.reset_index(drop=True, inplace=True)

        print_stats('ALLIGNMENT FILTER', f'{len(tmp_df)} images') 

        if commit:
            self.df = tmp_df.copy()
        else:
            return tmp_df
        
    def save_df(self,
                filepath: str = CONF.FILTERED_PATH):
        
        self.df.to_csv(filepath, sep="\t", index=False)


if __name__ == '__main__':
    explorer = RdrFilter()

    if CONF.filter_sequence:
        for filter in CONF.filter_sequence:
            getattr(explorer, filter)()
    else:
        explorer.load_df()
        explorer.latitude_filter()
        explorer.scale_filter()
        explorer.season_filter()
        explorer.cluster_filter()
        explorer.my_filter()
        explorer.allignment_filter()
        explorer.save_df()