#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Format all the data, then merge them into a single dataset."""


import pandas as pd
import os
current_dir = os.getcwd()
import sys
sys.path.append('..')
from utils import save_or_replace


OUTPUT_PATH = "../../data/02_converted_data/"

DATASET_PATH = "../../data/03_dataset/"


def merge():
    # get preprocessed data
    dico = dict()
    for file in os.listdir(OUTPUT_PATH):
        data = pd.read_csv(OUTPUT_PATH + file)
        dico[file] = data
    # get order
    order = ["activity", "action", "questionnaire", "quiz1", "quiz2", "quiz3",
             "quizf"]
    corresp_file = list()
    for name in order:
        for file in dico.keys():
            if name in file:
                corresp_file.append(file)
    # merge
    dataset = dico[corresp_file[0]]
    for i in range(1, len(corresp_file)):
        dataset = pd.merge(dataset, dico[corresp_file[i]], how="outer",
                           on=["st_id"])
    return dataset


def main():
    print("*********************** Formatting ************************", flush=True)
    os.chdir("formatting")
    os.system("python activity.py")
    os.system("python action.py")
    os.system("python questionnaire.py")
    os.system("python quizzes.py")
    os.chdir(current_dir)
    print("************************* Merging *************************", flush=True)
    dataset = merge()
    save_or_replace(dataset, "dataset", DATASET_PATH)


if __name__ == "__main__":
    main()
    print("All the previous converted data have been merged and saved in "
          "{}.".format(DATASET_PATH))
