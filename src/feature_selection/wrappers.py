#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrappers."""

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector, RFECV, RFE
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = "../../results/feature_selection/"

DATE = datetime.today().strftime('%Y%m%d')


def forward_elimination(data, target, k):
    model = LogisticRegression(solver="liblinear", random_state=42)
    selector = SequentialFeatureSelector(model, n_features_to_select=k,
                                         direction="forward")
    selector.fit(data, target)
    fs = np.array(data.columns)[selector.get_support()]
    print("Features selected by forward sequential selection:", ", ".join(fs))


def backward_elimination(data, target, k):
    model = LogisticRegression(solver="liblinear", random_state=42)
    selector = SequentialFeatureSelector(model, n_features_to_select=k,
                                         direction="backward")
    selector.fit(data, target)
    fs = np.array(data.columns)[selector.get_support()]
    print("Features selected by backward sequential selection:", ", ".join(fs))

def rfe(data, target, k):
    model = LogisticRegression(solver="liblinear", random_state=42)
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(data, target)
    fs = np.array(data.columns)[rfe.get_support()]
    print("Features selected by RFE:", ", ".join(fs))


def rfecv(data, target):
    print("Recursive feature elimination with cross-validation performed.")
    model = LogisticRegression(solver='liblinear', random_state=42)
    min_features_to_select = 1
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
                  scoring="accuracy",
                  min_features_to_select=min_features_to_select)
    rfecv.fit(data, target)
    print("  - Optimal number of features : %d" % rfecv.n_features_)
    # Plot number of features vs. cross-validation scores
    plt.plot(range(min_features_to_select,
                       len(rfecv.grid_scores_) + min_features_to_select),
                 rfecv.grid_scores_, "b")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    # save plot
    file_name = DATE + "_rfecv.png"
    plt.savefig(OUTPUT_PATH + file_name)
    print(f"  - Plot {file_name} saved in {OUTPUT_PATH}.")
    plt.show()
    return rfecv.n_features_