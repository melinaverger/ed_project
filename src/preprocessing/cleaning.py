#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for cleaning data."""

import sys
sys.path.append('..')
from utils import save_or_replace


OUTPUT_PATH = "../../data/04_transformed_dataset/"


def handle_outliers_quiz_time(data, type_):
    df = data.copy()
    column = df["act_quiz" + type_ + "_time"]
    new_column = (column <= 1200)*column  # 20 min threshold
    new_mean = new_column[new_column > 0].mean()
    new_column = new_column.replace(0, new_mean)
    df["act_quiz" + type_ + "_time"] = new_column
    return df


def handle_outliers_age(dataset):
    df = dataset.copy()
    column = df["st_age"]
    new_column = (column >= 18)*column  # 18 yo threshold
    new_mean = new_column[new_column > 0].mean()
    new_column = new_column.replace(0, new_mean)
    df["st_age"] = new_column
    return df


def remove_null_completion(dataset):
    dataset = dataset[dataset["%_completion"] > 0]
    return dataset


def remove_absent_final_grade(dataset):
    dataset = dataset[dataset["final_grade"] >= 0]
    return dataset


def remove_incomplete_questionnaire(dataset):
    variables = ["st_age", "st_french", "st_ed_level", "st_ed_field",
                 "st_descr_pos", "st_descr_neg"]
    dataset = dataset[dataset[variables].isna().any(axis=1) == False]
    return dataset


def separate_text_variables(dataset):
    variables = ["st_descr_pos", "st_descr_neg"]
    text_variables = dataset[["st_id"] + variables]
    save_or_replace(text_variables, "text_features", OUTPUT_PATH)
    dataset = dataset.drop(variables, axis=1)
    return dataset


def cleaning(dataset):
    nrows_i = dataset.shape[0]
    dataset = remove_null_completion(dataset)
    dataset = remove_absent_final_grade(dataset)
    dataset = remove_incomplete_questionnaire(dataset)
    dataset = separate_text_variables(dataset)
    nrows_f = dataset.shape[0]
    nrows_removed = nrows_i - nrows_f
    print(f"{int(nrows_removed/nrows_i*100)}% missing values.")
    return dataset
