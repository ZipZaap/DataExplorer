import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from configs.config_parser import Config

from shapely.geometry import Polygon
from shapely import intersection_all

CONF = Config('configs/config.yaml')

def download_from_pds(filepaths: list, 
                      url: str, 
                      savedir: Path,
                      reload: bool = False):
    """
    Download files from the given URL if they do not exist locally or if reload is True.

    Args
    --------
        filepaths : list
            A list of PDS relative filepaths.

        url : str
            The base URL to download files from.

        savedir : Path
            The directory where downloaded files are saved.

        reload : bool
            Forces download even if the file exists locally.

    Raises
    --------
        Exception
            If an error occurs during the download process.
    """
    
    # savedir.mkdir(parents=True, exist_ok=True)

    for relative_path in filepaths:
        filename = relative_path.split('/')[-1]
        local_path = savedir / filename
        temp_path = savedir / f"{filename}.part"

        if not local_path.exists() or reload:
            try:
                head_response = requests.head(f'{url}/{relative_path}')

                if head_response.status_code == 200:
                    size = int(head_response.headers.get("Content-Length", 0))
                    
                    # Use stream for files > 10MB
                    use_stream = size > 1e7 
                    
                    # Open the connection
                    response = requests.get(f'{url}/{relative_path}', stream=use_stream)
                    response.raise_for_status() # Check for HTTP errors

                    # Write to the TEMPORARY path first
                    block_size = int(1e5) 
                    
                    if use_stream:
                        progress_bar = tqdm(total=size, unit='iB', unit_scale=True, desc=filename)
                        with temp_path.open('wb') as f:
                            for data in response.iter_content(block_size):
                                progress_bar.update(len(data))
                                f.write(data)
                        progress_bar.close()
                    else:
                        with temp_path.open('wb') as f:
                            f.write(response.content)

                    # If we reached this line, download was successful and complete.
                    # Rename temp file to final filename (Atomic operation)
                    temp_path.rename(local_path)
                    print(f"{filename} downloaded successfully.")

                else:
                    print(f"Failed to download {filename}. Status code: {head_response.status_code}")

            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                # Cleanup: Delete the partial file if it exists
                if temp_path.exists():
                    temp_path.unlink()
        else:
            print(f"{filename} already exists. Set `reload = True` to force download.")

def arc2psp(LON: pd.Series, 
            LAT: pd.Series
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert planetocentric coordinates to map coordinates.

    Args
    ----
        LON : pd.Series
            Series of longitudes.

        LAT : pd.Series
            Series of latitudes.

    Returns
    -------
        out : tuple[np.ndarray, np.ndarray]
            List of (x, y) map coordinates.
    """
    x = 2 * 3376.2 * np.tan(np.pi / 4 - np.radians(LAT / 2)) * np.sin(np.radians(LON)) * 1000
    y = -2 * 3376.2 * np.tan(np.pi / 4 - np.radians(LAT / 2)) * np.cos(np.radians(LON)) * 1000
    return np.round(x, 2), np.round(y, 2)

def utc2my(ot: str) -> int:
    """
    Convert a UTC datetime string to a Mars year.

    Args
    ----
        ot : str
            A UTC datetime string in the format '%Y-%m-%dT%H:%M:%S'.

    Returns
    -------
        mars_year : int
            The Mars year computed from the reference date.
    """
    reference_date = datetime(1955, 4, 11)
    obs_date = datetime.strptime(ot.strip(), '%Y-%m-%dT%H:%M:%S')
    days_since_reference = abs((reference_date - obs_date).days)
    mars_year = int(days_since_reference / 687)
    return mars_year

def is_sequence(sequence: np.ndarray, 
                min_length: int, 
                max_gap: int
                ) -> list[int]:
    """
    Identify all integers belonging to subsequences that meet the minimum length
    and do not exceed the maximum allowed gap between elements.
    
    Args
    ----
        sequence : np.ndarray
            An array of integers.
            
        min_length : int
            The minimum subsequence length required.
            
        max_gap : int
            The maximum difference allowed between neighbor integers to consider 
            them part of the same sequence. 
            0 = strictly consecutive (e.g., 1, 2).
            1 = allows skipping one number (e.g., 1, 3).

    Returns
    -------
        result : list
            A list containing the elements of valid subsequences.
    """
    if len(sequence) == 0:
        return []

    unique_seq = np.unique(sequence)
    
    # Calculate difference between neighbors
    # A gap of 0 implies the numbers are consecutive (diff is 1).
    # We break the sequence if the difference is greater than max_gap + 1.
    # e.g., max_gap=0 (strict) -> break if diff > 1
    # e.g., max_gap=1 (loose)  -> break if diff > 2
    breaks = np.where(np.diff(unique_seq) > max_gap + 1)[0] + 1
    
    # Split the array into chunks based on the breaks
    sub_sequences = np.split(unique_seq, breaks)
    
    # Filter chunks meeting min_length
    valid_chunks = [seq for seq in sub_sequences if len(seq) >= min_length]
    
    if not valid_chunks:
        return []
        
    return np.concatenate(valid_chunks).tolist()

def calculate_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the area of the image footprint using the Shoelace formula.

    Args
    ----
        df : pd.DataFrame
            DataFrame containing the image metadata. 
            Must include 'C1_X'...'C4_X' and 'C1_Y'...'C4_Y' columns.

    Returns
    -------
        df : pd.DataFrame
            A copy of the input dataframe with an additional 'AREA' column.
    """
    
    df = df.copy()

    x_cols = [f'C{i}_X' for i in range(1, 5)]
    y_cols = [f'C{i}_Y' for i in range(1, 5)]

    x = df[x_cols].values
    y = df[y_cols].values

    x_shifted = np.roll(x, -1, axis=1)
    y_shifted = np.roll(y, -1, axis=1)

    df['AREA'] = (0.5 * np.abs(np.sum(x * y_shifted - x_shifted * y, axis=1))).astype(int)

    return df

def optimize_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters images to find the stack with the maximum overlapping intersection area.
    
    This method groups images within each cluster by their time bin (derived from 'MY').
    It then utilizes a Branch and Bound algorithm to select exactly one image per 
    bin such that the intersection area of all selected images is maximized.

    The algorithm uses two heuristics to optimize search speed:
    1. Polygons within bins are sorted by area (largest first).
    2. Bins are sorted by cardinality (fewest options first).

    Args
    ----
        df : pd.DataFrame
            A DataFrame containing the following image metadata:
            - 'MY': Temporal identifier used to rank and bin images.
            - 'C1_X', 'C1_Y' ... 'C4_X', 'C4_Y': Coordinates for the 4 corners of the footprint.

    Returns
    -------
        df : pd.DataFrame
            A trimmed DataFrame containing 1 best image per bin (MY).

    Raises
    ------
        ValueError: 
            If the filtering results in an empty dataset (no valid intersections found).
    """

    # 1. Create and bin polygons
    bins = []
    for _, group in df.groupby('MY'):
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

    df = df.loc[best_result['selection']]
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise(ValueError("No valid intersections found"))

    return df

def print_stats(x: str, 
                y: int | str, 
                total_width: int = 66):
    """
    Print two integers side by side with dots in between to fill the specified total width.

    Args
    ----
        x : int
            Description to be printed on the left side.

        y : int | str
            Value to be printed on the right side.
            
        total_width : int
            The total width of the printed line, including integers and dots.
    """

    x_str = str(x)
    y_str = str(y)
    num_dots = total_width - len(x_str) - len(y_str)
    if num_dots < 1:
        num_dots = 1
    dots = '.' * num_dots
    print(f'{x_str} {dots} {y_str}')