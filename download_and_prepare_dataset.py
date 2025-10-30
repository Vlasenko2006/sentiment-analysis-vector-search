#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 10:25:11 2025

@author: andreyvlasenko
"""

import os
import requests
import zipfile




def download_and_prepare_dataset():
# 1. Set paths
    volume_path = "/Users/andreyvlasenko/tst/Request/my_volume/"
    zip_filename = "trainingandtestdata.zip"
    csv_original = "training.1600000.processed.noemoticon.csv"
    csv_renamed = "sentiment140.csv"

    zip_path = os.path.join(volume_path, zip_filename)
    csv_path = os.path.join(volume_path, csv_original)
    csv_renamed_path = os.path.join(volume_path, csv_renamed)

    # 2. Download the zip file
    if not os.path.exists(zip_path):
        print("Downloading dataset...")
        url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded to {zip_path}")
    else:
        print(f"Zip file already exists at {zip_path}")

    # 3. Extract the CSV file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        if csv_original not in zip_ref.namelist():
            raise FileNotFoundError(f"{csv_original} not found in the ZIP archive.")
        print("Extracting CSV file...")
        zip_ref.extract(csv_original, volume_path)
        print(f"Extracted {csv_original} to {csv_path}")

    # 4. Rename the CSV file (optional)
    if not os.path.exists(csv_renamed_path):
        os.rename(csv_path, csv_renamed_path)
        print(f"Renamed to {csv_renamed_path}")
    else:
        print(f"{csv_renamed_path} already exists.")

    # 5. Save the resulting path to path_db variable
    path_db = csv_renamed_path
    print(f"path_db = '{path_db}'")

    return path_db

path_db = download_and_prepare_dataset()
