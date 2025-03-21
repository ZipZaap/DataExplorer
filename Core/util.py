import os
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import geojson as gj
from tqdm import tqdm

from configs.config_parser import Config

CONF = Config('configs/config.yaml')


def download_from_pds(url: str, 
                      filepaths: list[str], 
                      savedir: str, 
                      reload: bool = False
                     ) -> None:
    """
    Download files from the given URL if they do not exist locally or if reload is True.

    Args:
        url (str): The base URL to download files from.
        filepaths (list[str]): A list of PDS relative filepaths.
        savedir (str): The directory where downloaded files are saved.
        reload (bool): Forces download even if the file exists locally.
    """
    os.makedirs(savedir, exist_ok=True)

    for relative_path in filepaths:
        filename = relative_path.split('/')[-1]
        local_path = os.path.join(savedir, filename)

        if not os.path.exists(local_path) or reload:
            head_response = requests.head(f'{url}/{relative_path}')

            if head_response.status_code == 200:
                size = int(head_response.headers.get('Content-Length'))

                if use_stream := size > 1e7:  # 10MB
                    response = requests.get(f'{url}/{relative_path}', stream=use_stream)
                    block_size = int(1e5)  # 100KB per chunk

                    progress_bar = tqdm(total=size, unit='iB', unit_scale=True)
                    with open(local_path, 'wb') as f:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            f.write(data)
                    progress_bar.close()
                else:
                    response = requests.get(f'{url}/{relative_path}', stream=use_stream)
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                print(f"{filename} downloaded successfully.")
            else:
                print(f"Failed to download {filename}. Status code: {head_response.status_code}")
        else:
            print(f"{filename} already exists. Set `reload = True` to force download.")


def arc2psp(LON: pd.Series, LAT: pd.Series) -> list[tuple[float, float]]:
    """
    Convert planetocentric coordinates to map coordinates.

    Args:
        LON (pd.Series): Series of longitudes.
        LAT (pd.Series): Series of latitudes.

    Returns:
        list[tuple[float, float]]: List of (x, y) map coordinates.
    """
    x = 2 * 3376.2 * np.tan(np.pi / 4 - np.radians(LAT / 2)) * np.sin(np.radians(LON)) * 1000
    y = -2 * 3376.2 * np.tan(np.pi / 4 - np.radians(LAT / 2)) * np.cos(np.radians(LON)) * 1000
    return np.round(x, 2), np.round(y, 2)


def utc2my(ot: str) -> int:
    """
    Convert a UTC datetime string to a Mars year.

    Args:
        ot (str): A UTC datetime string in the format '%Y-%m-%dT%H:%M:%S'.

    Returns:
        mars_year (int): The Mars year computed from the reference date.
    """
    reference_date = datetime(1955, 4, 11)
    obs_date = datetime.strptime(ot.strip(), '%Y-%m-%dT%H:%M:%S')
    days_since_reference = abs((reference_date - obs_date).days)
    mars_year = int(days_since_reference / 687)
    return mars_year


def is_sequence(sequence: np.ndarray, min_length: int) -> list:
    """
    Identify a subsequence of consecutive integers within the given sequence 
    that meets the minimum length requirement.

    Args:
        sequence (np.ndarray): An array of integers.
        min_length (int): The minimum consecutive sequence length required.

    Returns:
        result (list): A list containing the longest subsequence found that meets or exceeds min_length.
    """
    sorted_seq = sorted(sequence.tolist())    
    result = []
    start = 0

    for i in range(1, len(sorted_seq) + 1):
        if i == len(sorted_seq) or sorted_seq[i] != sorted_seq[i - 1] + 1:
            if i - start >= min_length:
                result.extend(sorted_seq[start:i])
            start = i
    return result


def print_stats(x, y, total_width=66):
    x_str = str(x)
    y_str = str(y)
    num_dots = total_width - len(x_str) - len(y_str)
    if num_dots < 1:
        num_dots = 1
    dots = '.' * num_dots
    print(f'{x_str} {dots} {y_str}')