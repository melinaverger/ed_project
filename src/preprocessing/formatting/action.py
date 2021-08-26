#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the action data to a more easily manipulated format,
without changing them."""

import pandas as pd
import sys
sys.path.append('../..')
import utils


DATA_PATH = "../../../data/01_raw_data/"

OUTPUT_PATH = "../../../data/02_converted_data/"


def rename_columns(data):
    data.rename(columns={"ID": "st_id",
                         "Nombre total d'actions": "st_nb_action"},
                inplace=True)
    return data


def preprocess_data():
    action_name = utils.get_file_or_error(DATA_PATH, "action")
    data = pd.read_csv(DATA_PATH + action_name)
    data = rename_columns(data)
    utils.save_or_replace(data, "action", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_data()
    print("Action data have been converted and saved in "
          "{}.".format(OUTPUT_PATH))
    