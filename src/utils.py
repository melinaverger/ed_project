#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common fonctions."""

import os
import sys
from datetime import datetime
import pandas as pd
import zipfile


DATASET_PATH = "../../data/03_dataset/"

DATE = datetime.today().strftime('%Y%m%d')


def get_file_or_error(path, name):
    for file in os.listdir(path):
        if name in file:
            file_name = file
    try:
        os.path.exists("{}{}".format(path, file_name))
    except NameError:
        print("Error: {} file is not present.".format(name))
        sys.exit(1)
    return file_name


def save_or_replace(data, name, output_path):
    # remove if existing file
    for file in os.listdir(output_path):
        if name in file:
            os.remove(output_path + file)
    # save new file
    data.to_csv(output_path + DATE + "_" + name + ".csv", index=False)


def load_dataset(path=DATASET_PATH):
    if (len(os.listdir(path)) == 1) or len(os.listdir(path)) == 2:
        file_name = get_file_or_error(path, "dataset")
    elif len(os.listdir(path)) > 2:
        raise ValueError("Too many file in directory (must have 1).")
    else:
        print("Error: dataset file is not present.")
        sys.exit(1)
    dataset = pd.read_csv(path + file_name)
    return dataset


def load_open(name):
    if name == "cnp":
        zf = zipfile.ZipFile("../../data/open_cnp/" + "CNPC_1401-1509_DI_v1_1_2016-03-01.csv.zip") 
        dataset = pd.read_csv(zf.open("CNPC_1401-1509_DI_v1_1_2016-03-01.csv"), low_memory=False)
        #dataset = pd.read_csv("../../data/open_cnp/" + "CNPC_1401-1509_DI_v1_1_2016-03-01.csv", low_memory=False)
    elif name == "olc":
        dataset = pd.read_csv("../../data/open_olc/saved_data/" + "data_merged.csv")
    else:
        raise ValueError("load_open can only take 'cnp' or 'olc' as argument.")
    return dataset
