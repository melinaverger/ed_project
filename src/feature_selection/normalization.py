#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normalizing the dataset."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize(x):
    feature_names = x.columns
    scaler = MinMaxScaler()
    x = pd.DataFrame(scaler.fit_transform(x))
    x.columns = feature_names
    return x
