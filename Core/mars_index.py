import os
import pandas as pd
from pathlib import Path
from dbscan import DBSCAN # type: ignore[reportAttributeAccessIssue]

from Core.mars_plotter import MapPlotter
from configs.config_parser import Config
from configs.validators import validate_and_log
from .util import arc2psp, utc2my, is_sequence, download_from_pds,\
                    print_stats, calculate_area, optimize_overlap

CONF = Config('configs/config.yaml')

class ImageIndex():
    
    def __init__(self, 
                 url: str = CONF.URL, 
                 idx_path: list = CONF.PATH2IDX, 
                 idx_savedir: Path = CONF.IDX_DIR,
                 rdr_savedir: Path = CONF.RDR_DIR,
                 jpg_savedir: Path = CONF.PREVIEW_DIR,
                 csv_savedir: Path = CONF.CSV_DIR,
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
            
            rdr_savedir : Path
                Directory to save the RDR files.
            
            csv_savedir : Path
                Directory to save the .csv files.

            columns : list[str]
                List of columns to keep in the dataframe.
            
            red : bool
                Whether to keep only the RED images (single-channel).
        """
        # create a plotter
        self.plotter = MapPlotter()

        # initialize paths & local configuration dictionary
        self.url = url
        self.idx_savedir = idx_savedir
        self.rdr_savedir = rdr_savedir
        self.jpg_savedir = jpg_savedir
        self.csv_savedir = csv_savedir
        self.local_conf = {}
        
        # download index files if not present
        download_from_pds(idx_path, url, idx_savedir)

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
        self.tmp_df['CLUSTER'] = pd.factorize(self.tmp_df['CLUSTER'], sort=True)[0] + 1
        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError("No clusters found. Try changing the clustering parameters.")
        else:
            print_stats('DENSITY FILTER', 
                f'{len(self.tmp_df.CLUSTER.unique())} locations / {len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()


    @validate_and_log
    def temporal_filter(self, 
                        mys: list[int] = CONF.mys, 
                        min_years: int = CONF.min_years, 
                        max_gap: int = CONF.max_gap,
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
                valid_years_subset = is_sequence(years, min_years, max_gap)

            for y in valid_years_subset:
                valid_clusters_map.append({'CLUSTER': cluster, 'MY': y})

        # 3. Apply Filter: Inner merge automatically filters rows that don't match both Cluster and MY
        if valid_clusters_map:
            self.tmp_df = self.tmp_df.merge(pd.DataFrame(valid_clusters_map), on=['CLUSTER', 'MY'], how='inner')
            self.tmp_df['CLUSTER'] = pd.factorize(self.tmp_df['CLUSTER'], sort=True)[0] + 1
            self.tmp_df.reset_index(drop=True, inplace=True)

            print_stats('TEMPORAL FILTER', 
                f'{len(self.tmp_df.CLUSTER.unique())} locations / {len(self.tmp_df)} images')
        else:
            raise ValueError("No clusters found. Try changing the Mars Year filtering parameters.")

        if commit:
            self.df = self.tmp_df.copy()


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

        self.tmp_df['CLUSTER'] = pd.factorize(self.tmp_df['CLUSTER'], sort=True)[0] + 1
        self.tmp_df.reset_index(drop=True, inplace=True)

        if self.tmp_df.empty:
            raise ValueError("No clusters found. Try changing the keyword filtering parameters.")
        else:
            print_stats('KEYWORD FILTER', 
                f'{len(self.tmp_df.CLUSTER.unique())} locations / {len(self.tmp_df)} images')

        if commit:
            self.df = self.tmp_df.copy()


    def show_on_map(self,
                    target: str, 
                    engine: str = "pygmt", 
                    color: str = "purple",
                    title: str | None = None):
        """
        Visualize the filtered data on a map using the specified engine.

        Args
        ----
            target : str
                The type of data to visualize. 
                Must be one of: 'img_footprint', 'img_centroid', 'cluster'.

            engine : str
                The visualization engine to use. One of: 'pygmt', 'qgis'.
                Defaults to "pygmt".

            color : str
                Color of the plotted elements. Defaults to "purple".

            title : str | None
                Title of the plot. If provided, the plot is saved to a file.
                If None, the plot is displayed interactively (PyGMT only).

        Raises
        ------
            RuntimeError
                If the filtered dataframe is empty or None.

            ValueError
                If an unsupported target is provided.

            RuntimeError
                If 'cluster' target is selected but clustering has not been performed.
        """

        if self.tmp_df is None or self.tmp_df.empty:
            raise RuntimeError("Dataframe is empty, please apply filters before visualization")
        else:
            plot_df = self.tmp_df.copy()

        if target not in {'img_footprint', 'img_centroid', 'cluster'}:
            raise ValueError("Unsupported target! Choose one of: `img_footpirnt`, `img_centroid`, `cluster`.")
        elif target == 'cluster':
            if 'CLUSTER' not in plot_df.columns:
                raise RuntimeError("Cluster centroids can't be mapped before cluster_filter() is applied")
            plot_df = plot_df.groupby('CLUSTER')[['CTR_LON', 'CTR_LAT', 'CTR_X', 'CTR_Y']].mean().reset_index()

        self.plotter.show_on_map(plot_df, target, color, engine, title)


    def show_preview(self, cluster_id: int):
        """
        Display a grid of thumbnail previews for a specific cluster.

        This method downloads the thumbnail images for the specified cluster 
        and displays them organized by Mars Year using the MapPlotter.

        Args
        ----
            cluster_id : int
                The ID of the cluster to visualize.

        Raises
        ------
            RuntimeError
                If clustering has not yet been performed (i.e., density_filter() was not called).
        """

        if 'CLUSTER' not in self.df.columns:
            raise RuntimeError("Can't show a preview of a cluster before density_filter() is applied")
        
        # get pds relative thumbnail paths
        preview_df = self.df[self.df.CLUSTER == cluster_id].copy()
        filepaths = preview_df.FILE_NAME_SPECIFICATION.tolist()
        filepaths = [f"EXTRAS/{p.replace('.JP2', '.thumb.jpg')}" for p in filepaths]

        # download thumbnails
        download_from_pds(filepaths, self.url, self.jpg_savedir, False)

        # display thumbnails arragned by Mars Year
        self.plotter.show_preview(preview_df, self.jpg_savedir)


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
                        cluster_id: int,
                        exclude: list[str] = [],
                        allign: bool = True,
                        reload: bool = False):
        """
        Download full-scale images for a specific cluster from the PDS server.

        This method filters the dataframe for a given `cluster_id`, optionally
        excluding specified product IDs. It can also perform an alignment step
        to select the stack of images with the maximum overlapping area.
        
        A CSV file containing metadata of the images to be downloaded is saved,
        and then the full-scale .JP2 products are retrieved.

        Args
        ----
            cluster_id : int
                The ID of the cluster to download images for.

            exclude : list[str], optional
                A list of 'PRODUCT_ID's to exclude from the download. 

            allign : bool, optional
                If True, optimizes the image selection within the cluster to 
                maximize the overlapping area. 

            reload : bool, optional
                If True, forces download even if the file already exists locally. 

        Raises
        ------
            RuntimeError
                If clustering has not yet been performed (i.e., density_filter() was not called).
        """

        if 'CLUSTER' not in self.df.columns:
            raise RuntimeError("Can't download by cluster_id before cluster_filter() is applied")
        
        self.tmp_df = self.df[self.df['CLUSTER'] == cluster_id].copy()
        
        if exclude:
            self.tmp_df = self.tmp_df[~self.tmp_df['PRODUCT_ID'].isin(exclude)]

        if allign:
            self.tmp_df = optimize_overlap(self.tmp_df)
        
        # calculate area
        self.tmp_df = calculate_area(self.tmp_df)
        self.tmp_df.to_csv(os.path.join(self.csv_savedir, f"cid_{cluster_id}.csv"), sep=",", index=False)

        # get pds relative filepaths
        filepaths = self.tmp_df.FILE_NAME_SPECIFICATION.tolist()

        # retrieve full-scale .jp2 products
        download_from_pds(filepaths, self.url, self.rdr_savedir, reload)



if __name__ == '__main__':
    index = ImageIndex()

    if CONF.filter_sequence:
        for filter in CONF.filter_sequence:
            getattr(index, filter)()
    else:
        raise ValueError('Define a non-empty filter_sequence in `configs.yaml`.')