#!python
"""Download script for the MovieLens 100K dataset.

See --help option for more information on usage.

Usage:
    python download.py
"""
import argparse
import os
import pathlib
import zipfile

import tqdm
import requests

CHUNK_SIZE = 8192
DATASET_URL = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
FILE_NAME = 'ml-100k.zip'


class Args:
    """Container class for command line arguments.
    """
    target_dir: str
    unzip: bool


def parse_args() -> Args:
    """Parses command line arguments and returns them in namespace
    that conforms to the `Args` interface.

    Returns:
        Args: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--target-dir',
        dest='target_dir',
        help='The directory to download the files into. Defaults to ./data/raw',
        default='./data/raw')
    parser.add_argument('--no-unzip',
                        dest='unzip',
                        action='store_false',
                        help='Do not unzip the downloaded file.')
    args = parser.parse_args()
    return args


def main():
    """Entry point of the script.
    """
    args = parse_args()
    curdir = pathlib.Path(os.getcwd())
    target_dir = curdir / args.target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / FILE_NAME
    with requests.get(DATASET_URL, stream=True, timeout=10) as request:
        request.raise_for_status()
        total_size = float(request.headers.get('Content-Length', 0))
        bar_format = "{desc}: {percentage:.0f}%|{bar}| {n:.2f}/{total:.2f} MB [{elapsed}<{remaining}]"
        with tqdm.tqdm(total=total_size / 1e6 + 1e-6,
                       bar_format=bar_format) as pbar, open(target_file,
                                                            'wb') as file:
            for chunk in request.iter_content(chunk_size=CHUNK_SIZE):
                file.write(chunk)
                pbar.update(len(chunk) / 1e6)
    if args.unzip:
        with zipfile.ZipFile(target_file) as archive:
            archive.extractall(target_dir)
        os.remove(target_file)


if __name__ == '__main__':
    main()
