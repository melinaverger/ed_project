#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select target variable type."""

def regression(dataset):
    target_col = "final_grade"
    dataset["target"] = dataset[target_col]
    dataset.drop(target_col, axis=1, inplace=True)
    print("Creation of a regression target variable.", flush=True)
    return dataset


def binary_classification(dataset):
    target_col = "final_grade"
    success_threshold = 6*1.11
    dataset["target"] = dataset[target_col] >= success_threshold
    dataset["target"].replace(to_replace=[True, False], value=[1, 0], inplace=True)
    dataset = dataset.drop(target_col, axis=1)
    print(f"Creation of a binary class target variable with {round(success_threshold, 2)} threshold.", flush=True)
    print(f"{round(len(dataset['target'][dataset['target'] == 1])/len(dataset['target'])*100, 2)}% of success (1) in target variable.", flush=True)
    return dataset


def target(dataset, task):
    dataset = dataset.drop("st_id", axis=1)
    if task == "regression":
        return regression(dataset)
    elif task == "binary classification":
        return binary_classification(dataset)
    else:
        raise ValueError("target() takes only 'regression' or 'binary classification' as task argument.")