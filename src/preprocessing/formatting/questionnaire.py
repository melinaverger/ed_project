#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the questionnaire data to a more easily manipulated format,
without changing them."""

import pandas as pd
import sys
sys.path.append('../..')
import utils


DATA_PATH = "../../../data/01_raw_data/"

OUTPUT_PATH = "../../../data/02_converted_data/"


def rename_columns(data):
    data.rename(columns={"ID": "st_id"}, inplace=True)
    for column in data.columns:
        if "Q" in column:
            tmp = column.split("_")[1:]
            new_column = "_".join(tmp)
            data.rename(columns={column: new_column}, inplace=True)
    return data


def preprocess_data():
    questionnaire_name = utils.get_file_or_error(DATA_PATH, "questionnaire")
    data = pd.read_csv(DATA_PATH + questionnaire_name)
    data = rename_columns(data)
    utils.save_or_replace(data, "questionnaire", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_data()
    print("Questionnaire data have been converted and saved in "
          "{}.".format(OUTPUT_PATH))
