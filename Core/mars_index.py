import os
import itertools
import pandas as pd
from pathlib import Path
from dbscan import DBSCAN # type: ignore[reportAttributeAccessIssue]
from functools import reduce
from shapely.geometry import Polygon
from shapely import intersection_all

from configs.config_parser import Config
from configs.validators import validate_and_log
from .util import arc2psp, utc2my, is_sequence, download_from_pds, print_stats

CONF = Config('configs/config.yaml')

class ImageIndex():
    
    def __init__(self, 
                 url: str = CONF.URL, 
                 idx_path: list = CONF.PATH2IDX, 
                 idx_savedir: Path = CONF.IDX_DIR,
                 columns: list[str] = CONF.columns,
                 red: bool = True):
        """
        Initialize RdrFilter instance. Load the dataframe from label and tab files. 
        Keep only the specified columns. Discard the images that are not single-channel RED. 
        Convert the areocentric co-ordinates to 2D polar stereographic.

        Args
        ----
            url : str
                URL of the PDS server.

            idx_paths : dict
                Dictionary of index file paths.

            idx_savedir : Path
                Directory to save the index files.

            columns : list[str]
                List of columns to keep in the dataframe.
            
            red : bool
                Whether to keep only the RED images (single-channel).
        """

        # initialize paths & local configuration dictionary
        self.url = url
        self.idx_savedir = idx_savedir
        self.local_conf = {}
        
        # download index files if not present
        download_from_pds(url, idx_path, idx_savedir)

        # load .tab
        lbl_path = idx_savedir / "RDRCUMINDEX.LBL"
        tab_path = idx_savedir / "RDRCUMINDEX.TAB"
        with lbl_path.open('r') as f:
            labels = f.read().splitlines() 
        labels = [label.split()[2] for label in labels if 'NAME' in label]
        self.df = pd.read_csv(tab_path, names=labels)

        # keep only the useful columns; clean the PRODUCT_ID string.
        self.df = self.df[columns]
        self.df['PRODUCT_ID'] = self.df['PRODUCT_ID'].str.strip()
        self.df['FILE_NAME_SPECIFICATION'] = self.df['FILE_NAME_SPECIFICATION'].str.strip()

        # keep only the RED images (1 channel)
        if red:
            self.df = self.df[self.df['PRODUCT_ID'].str.contains('RED')]

        # convert the (LAT, LON) areocentric [ARC] co-oridantes to (X, Y) 2D polar stereographic [PSP]
        for i in range(1, 5):
            self.df.rename(columns={f'CORNER{i}_LONGITUDE':f'C{i}_LON', f'CORNER{i}_LATITUDE': f'C{i}_LAT'}, inplace=True)
            self.df[f'C{i}_X'], self.df[f'C{i}_Y'] = arc2psp(self.df[f'C{i}_LON'], self.df[f'C{i}_LAT'])
            
        # calculate image centroids
        for coor in ['LON', 'LAT', 'X', 'Y']:
            self.df[f'CTR_{coor}'] = self.df[[f'C{i}_{coor}' for i in range(1, 5)]].mean(axis=1)

        self.df.reset_index(drop=True, inplace=True)

        print_stats('DATASET SIZE', f'{len(self.df)} images')


    @validate_and_log
    def latitude_filter(self, 
                        min_lat: int = CONF.min_lat,
                        commit: bool = True):
        """
        Keep only the images above the specified minimum latitude. 
        Validate and log the parameters into the `local_conf` dictionary. 

        Args
        ----
            min_lat : int
                Minimum latitude value.

            commit : bool
                Whether to commit the changes to the main dataframe.

        Raises
        ------
            ValueError
                If the minimum latitude is out of range.

            ValueError
                If no images are found above the specified latitude.

        Returns
        -------
            self.tmp_df :pd.DataFrame
                The filtered dataframe.
        """

        self.tmp_df = self.df.copy()
        self.tmp_df = self.tmp_df[(self.tmp_df[[f'C{i}_LAT' for i in range(1,5)]] > min_lat).any(axis=1)]
        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError(f"No images found above the specified latitude: {min_lat}.")
        else:
            print_stats('LATITUDE FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    @validate_and_log
    def scale_filter(self, 
                     scale: float = CONF.scale,
                     commit: bool = True):
        """
        Filter images by map scale.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args
        ----
            scale : float
                Desired map scale. Must be one of: 0.25, 0.5, 1.

            commit : bool
                Whether to commit the changes to the main dataframe.

        Raises
        ------
            ValueError
                If an unsupported scale is provided.

            ValueError
                If no images are found for the specified scale.

        Returns
        -------
            self.tmp_df : pd.DataFrame
                The filtered dataframe.
        """
        
        self.tmp_df = self.df.copy()
        self.tmp_df = self.tmp_df[self.tmp_df['MAP_SCALE'] == scale]
        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError(f"No images found for the specified scale: {scale}.")
        else:
            print_stats('SCALE FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    @validate_and_log
    def season_filter(self, 
                      season: str = CONF.season,
                      commit: bool = True):
        """
        Filter images based on the specified season.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args
        ----
            season : str
                Season string to filter images by. 
                Must be of `<hemisphere> <season>` format"

            commit : bool
                Whether to commit the changes to the main dataframe.

        Raises
        ------
            ValueError
                If an unsupported season is provided.

            ValueError
                If no images are found for the specified season.

        Returns
        -------
            self.tmp_df : pd.DataFrame
                The filtered dataframe.
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
        else:
            raise ValueError("Invalid season! Use `<hemisphere> <season>` format")

        self.tmp_df = self.df.copy()
        self.tmp_df = self.tmp_df[(self.tmp_df['SOLAR_LONGITUDE'] > min_sol_long) & 
                        (self.tmp_df['SOLAR_LONGITUDE'] < max_sol_long)]
        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError(f"No images found for the specified season: {season}.")
        else:
            print_stats('SEASON FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    @validate_and_log
    def density_filter(self,
                       min_samples: int = CONF.min_samples, 
                       epsilon: int = CONF.epsilon,
                       commit: bool = True):
        """
        Perform clustering on image centroids using the specified algorithm.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args
        ----
            min_samples : int
                Minimum numbers of points that can form a cluster.

            epsilon : int
                Epsilon value (neighborhood size) for DBSCAN / HDBSCAN.
                This should be provided in the distance units of the input data.

            commit : bool
                Whether to commit the changes to the main dataframe.

        Raises
        ------
            ValueError
                If min_samples < 2.

            ValueError
                If no clusters are found with the specified parameters.

        Returns
        -------
            self.tmp_df : pd.DataFrame
                The filtered dataframe.
        """

        self.tmp_df = self.df.copy()

        labels, _ = DBSCAN(
            self.tmp_df[['CTR_X', 'CTR_Y']].values, 
            min_samples=min_samples, 
            eps=epsilon
        )

        # create a new CLUSTER column & discard outlier images
        self.tmp_df['CLUSTER']  = labels
        self.tmp_df = self.tmp_df[self.tmp_df['CLUSTER'] != -1]
        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError("No clusters found. Try changing the clustering parameters.")
        else:
            print_stats('CLUSTER FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    @validate_and_log
    def temporal_filter(self, 
                        min_years: int = CONF.min_years, 
                        mys: list[int] = CONF.mys, 
                        seq: bool = CONF.consecutive,
                        commit: bool = True):
        """
        Filter clusters based on the sequence of unique Mars Year (MY) it contains.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args
        ----
            min_years : int
                Minimum number of unique Mars Years to be present in a cluster.

            mys : list[int]
                Specific Mars Years that must be present. Overwrites `min_years`.

            seq : bool
                Whether the years must be consecutive.

            commit : bool
                Whether to commit the changes to the main dataframe.

        Raises
        ------
            ValueError
                If min_years < 2.

            ValueError
                If no clusters satisfy the Mars Year filtering criteria.

            RuntimeError
                If the current filter is applied before cluster_filter().

        Returns
        -------
            self.tmp_df : pd.DataFrame
                The filtered dataframe.
        """
        if 'CLUSTER' not in self.df.columns:
            raise RuntimeError("Current filter can't be applied before cluster_filter()")

        # convert UTC time to Mars Year (MY). current range of MY is 27-35*
        self.tmp_df = self.df.copy()
        self.tmp_df['MY'] = self.tmp_df['OBSERVATION_START_TIME'].apply(utc2my)
        self.tmp_df.drop(['OBSERVATION_START_TIME'], axis=1, inplace=True)

        # 1. Summarize: Get unique years per cluster
        cluster_summary = self.tmp_df.groupby('CLUSTER')['MY'].unique()
        
        # 2. Filter Logic: Identify valid (Cluster, Year) pairs
        valid_clusters_map = []
        for cluster, years in cluster_summary.items():
            valid_years_subset = []
 
            if mys:
                if set(mys).issubset(years):
                    valid_years_subset = mys
    
            elif len(years) >= min_years:
                if seq:
                    valid_years_subset = is_sequence(years, min_years)
                else:
                    valid_years_subset = years.tolist()
        
            for y in valid_years_subset:
                valid_clusters_map.append({'CLUSTER': cluster, 'MY': y})

        # 3. Apply Filter: Inner merge automatically filters rows that don't match both Cluster and MY
        if valid_clusters_map:
            self.tmp_df = self.tmp_df.merge(pd.DataFrame(valid_clusters_map), on=['CLUSTER', 'MY'], how='inner')
            print_stats('YEAR FILTER', f'{len(self.tmp_df)} images')
        else:
            raise ValueError("No clusters found. Try changing the Mars Year filtering parameters.")

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    @validate_and_log
    def keyword_filter(self, 
                       keywords: list[str] = CONF.keywords,
                       commit: bool = True):
        """
        Filter clusters based on the presence of specific keywords in the RATIONALE_DESC column.
        Validate and log the parameters into the `local_conf` dictionary. 

        Args
        ----
            keywords : list[str]
                List of keywords to filter by.

            commit : bool
                Whether to commit the changes to the main dataframe.

        Raises
        ------
            ValueError
                If no clusters satisfy the keyword filtering criteria.

            RuntimeError
                If the current filter is applied before cluster_filter().

        Returns
        -------
            self.tmp_df : pd.DataFrame
                The filtered dataframe.
        """

        if 'CLUSTER' not in self.df.columns:
            raise RuntimeError("Current filter can't be applied before cluster_filter()")

        self.tmp_df = self.df.copy()

        pattern = r'(?i)(?:' + '|'.join(keywords)  + r')'
        self.tmp_df = self.tmp_df.groupby("CLUSTER").filter(
            lambda group: group["RATIONALE_DESC"].str.contains(pattern).any()
        )

        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError("No clusters found. Try changing the keyword filtering parameters.")
        else:
            print_stats('KEYWORD FILTER', f'{len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    @validate_and_log
    def alignment_filter(self, commit: bool = True):
        """
        Filters images to find the stack with the maximum overlapping intersection area.
        
        This method groups images within each cluster by their time bin (derived from 'MY').
        It then utilizes a Branch and Bound algorithm to select exactly one image per 
        bin such that the intersection area of all selected images is maximized.

        The algorithm uses two heuristics to optimize search speed:
        1. Polygons within bins are sorted by area (largest first).
        2. Bins are sorted by cardinality (fewest options first).

        Prerequisites
        -------------
        The dataframe must contain:
        - 'CLUSTER': Grouping identifier.
        - 'MY': Temporal identifier used to rank and bin images.
        - 'C1_X', 'C1_Y' ... 'C4_X', 'C4_Y': Coordinates for the 4 corners of the footprint.

        Args
        ----
            commit (bool): 
                Whether to commit the changes to the main dataframe.

        Returns
        -------
            pd.DataFrame: 
                The filtered dataframe containing only the selected images that form 
                the best geometric stack.

        Raises
        ------
            RuntimeError: 
                If the 'CLUSTER' column is missing (filtering applied out of order).

            ValueError: 
                If the filtering results in an empty dataset (no valid intersections found).
        """
        
        if 'CLUSTER' not in self.df.columns:
            raise RuntimeError("Current filter can't be applied before cluster_filter()")
        
        self.tmp_df = self.df.copy()

        filtered_ids = []
        for cluster in self.tmp_df.CLUSTER.unique():
            cluster_df = self.tmp_df[self.tmp_df.CLUSTER == cluster].copy()
            cluster_df['bin_id'] = cluster_df.MY.rank(method='dense').astype(int)

            # 1. Create and bin polygons
            bins = []
            for _, group in cluster_df.groupby('bin_id'):
                current_bin = []
                for poly_id, row in group.iterrows():
                    coords = [(row[f'C{i}_X'], row[f'C{i}_Y']) for i in range(1, 5)]
                    current_bin.append((Polygon(coords), poly_id))
                
                # 2. Sort polygons within each bin by area (Heuristic: largest first)
                current_bin.sort(key=lambda x: x[0].area, reverse=True)
                bins.append(current_bin)

            # 3. Sort bins by number of elements (Heuristic: fewest options first)
            bins.sort(key=len)

            # 4. Quick greedy search to set a good starting point
            try:
                greedy_start = intersection_all([b[0][0] for b in bins]).area
                greedy_selection = [b[0][1] for b in bins]
                best_result = {'area': greedy_start, 'selection': greedy_selection}
            except Exception:
                best_result = {'area': 0.0, 'selection': []}

            # 5. Find the best intersection using "Branch and Bound" algorithm
            def _search(depth, current_intersection, current_selection):
                # --- BOUNDING (PRUNING) STEP ---
                # If we have a valid intersection context (depth > 0)
                if current_intersection is not None:
                    
                    # If the current intersection is empty, this branch is dead.
                    if current_intersection.is_empty:
                        return

                    # PRUNING: If current area is already <= the best area found, this branch is also dead.
                    if current_intersection.area <= best_result['area']:
                        return

                # --- BASE CASE ---
                # We have selected one polygon from every bin
                if depth == len(bins):
                    current_area = current_intersection.area
                    if current_area > best_result['area']:
                        best_result['area'] = current_area
                        best_result['selection'] = sorted(list(current_selection))
                    return

                # --- BRANCHING (RECURSIVE) STEP ---
                # Iterate through candidates in the current bin
                for poly, poly_id in bins[depth]:
                    
                    # If the candidate polygon itself is smaller than best_area,
                    # the intersection will definitely be smaller too. Skip it.
                    if poly.area <= best_result['area']:
                        continue
                    
                    # Calculate new intersection
                    if current_intersection is None:
                        new_intersection = poly
                    else:
                        new_intersection = current_intersection.intersection(poly)
                
                    # Recursive call
                    _search(depth + 1, new_intersection, current_selection + [poly_id])

            # run the search
            _search(0, None, [])
            filtered_ids.extend(best_result['selection'])

        self.tmp_df = self.tmp_df.loc[filtered_ids]

        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError("No clusters found. Try changing the allignment filtering parameters.")
        else:
            print_stats('ALLIGNMENT FILTER', f'{len(self.tmp_df)} images') 

        if commit:
            self.df = self.tmp_df.copy()

        return self.tmp_df


    def save_df(self, filename: str = 'FILTERED'):
        """
        Save the current dataframe as a .TAB file in the specified directory.
        Args
        ----
            filename : str
                Name of the output file.
        """

        savepath = os.path.join(self.idx_savedir, f"{filename}.TAB")
        self.df.to_csv(savepath, sep="\t", index=False)


    def download_images(self, 
                        product_id: list[str] | None = None,
                        cluster_id: int | None = None,
                        savedir: Path = CONF.RDR_DIR,
                        reload: bool = False):
        """
        Download the filtered images from the PDS server. If `product_id` is provided, 
        download images based on the product ID, else download based on the cluster ID.
  
        Args
        ----
            product_id : list[str]
                List of product IDs to download.

            cluster_id : int
                Cluster ID to download.

            reload : bool
                Forces download even if the file exists locally.

        Raises
        ------
            ValueError
                If neither `product_id` nor `cluster_id` is provided.

            RuntimeError
                If retriveal by `cluster_id` is attempted before cluster_filter() is applied.
        """

        if product_id:
            rdr_paths = self.df[self.df['PRODUCT_ID'].isin(product_id)]['FILE_NAME_SPECIFICATION'].tolist()
        elif cluster_id:
            if 'CLUSTER' not in self.df.columns:
                raise RuntimeError("Can't download by cluster_id before cluster_filter() is applied")
            else:
                rdr_paths = self.df[self.df['CLUSTER'] == cluster_id]['FILE_NAME_SPECIFICATION'].tolist()
        else:
            raise ValueError("At least one of `product_id` or `cluster_id` must be provided")

        download_from_pds(self.url, rdr_paths, savedir, reload)


if __name__ == '__main__':
    index = ImageIndex()

    if CONF.filter_sequence:
        for filter in CONF.filter_sequence:
            getattr(index, filter)()
    else:
        raise ValueError('Define a non-empty filter_sequence in `configs.yaml`.')