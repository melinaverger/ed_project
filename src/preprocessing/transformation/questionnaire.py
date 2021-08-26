#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transform questionnaire variables."""

import sys
sys.path.append('..')
from cleaning import handle_outliers_age


def transformation(dataset):
    variables = ["st_gender", "st_agreement"]
    dataset.drop(variables, axis=1, inplace=True)
    dataset = handle_outliers_age(dataset)
    return dataset
