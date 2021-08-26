#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove low-variance features."""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


THRESHOLD = 0


def extra(l1, l2):
    l1, l2 = list(l1), list(l2)
    if len(l1) != len(l2):
        extra = list()
        if len(l2) > len(l1):
            l1, l2 = l2[:], l1[:]
        for i in range(len(l1)):
            if l1[i] not in l2:
                extra.append(l1[i])
        return extra


def low_variance_removal(x):
    feature_names = x.columns
    selector = VarianceThreshold(threshold=THRESHOLD)
    selector.fit(x)
    retained_features = feature_names[selector.get_support(indices=True)]
    x = pd.DataFrame(selector.transform(x))
    x.columns = retained_features
    rejected = extra(feature_names, retained_features)
    print(f"Threshold: {THRESHOLD}")
    if len(rejected) > 0:
        print("Feature(s) removed:", ", ".join(rejected))
    else:
        print("No features removed.")
    return x
