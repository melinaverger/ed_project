#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform balancing techniques for D1, D2 and D3."""

import prediction
import pandas as pd
import sys
sys.path.append('..')
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE 


DATA_PATH = "../../data/06_common_data/"
MODEL = "decision tree"


d1 = pd.read_csv(DATA_PATH + "d1.csv")
d2 = pd.read_csv(DATA_PATH + "d2.csv")
d3 = pd.read_csv(DATA_PATH + "d3.csv")


def no_sampling(data):
    print("----- No sampling -----")
    prediction.pred(data, MODEL)


def downsampling(data):
    counts = data["target"].value_counts()
    nb_0 = counts[0]
    nb_1 = counts[1]
    # separate majority and minority classes
    if nb_0 > nb_1:
        thresh = nb_1
        df_majority = data[data.target==0]
        df_minority = data[data.target==1]
    else:
        thresh = nb_0
        df_majority = data[data.target==1]
        df_minority = data[data.target==0]
    # downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,
                                     n_samples=thresh,
                                     random_state=0)
    # combine majority class with upsampled minority class
    data = pd.concat([df_majority_downsampled, df_minority])
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    print("----- Downsampling -----")
    prediction.pred(data, MODEL)


def upsampling(data):
    counts = data["target"].value_counts()
    nb_0 = counts[0]
    nb_1 = counts[1]
    # separate majority and minority classes
    if nb_0 > nb_1:
        thresh = nb_0
        df_majority = data[data.target==0]
        df_minority = data[data.target==1]
    else:
        thresh = nb_1
        df_majority = data[data.target==1]
        df_minority = data[data.target==0]
    # downsample majority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,
                                     n_samples=thresh,
                                     random_state=0)
    # combine majority class with upsampled minority class
    data = pd.concat([df_majority, df_minority_upsampled])
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    print("----- Upsampling -----")
    prediction.pred(data, MODEL)


def up_and_dowsnsampling(data, type_):
    thresh = data.shape[0]//2
    counts = data["target"].value_counts()
    nb_0 = counts[0]
    nb_1 = counts[1]
    # separate majority and minority classes
    if nb_0 > nb_1:
        df_majority = data[data.target==0]
        df_minority = data[data.target==1]
    else:
        df_majority = data[data.target==1]
        df_minority = data[data.target==0]     
    # upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,
                                     n_samples=thresh,
                                     random_state=0)
    # downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,
                                     n_samples=thresh,
                                     random_state=0)
    # combine majority class with upsampled minority class
    data = pd.concat([df_majority_downsampled, df_minority_upsampled])
    # display new class counts
    data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    data.to_csv(DATA_PATH + "{}_balanced.csv".format(type_), index=False)
    print("----- Up and Downsampling ----")
    prediction.pred(data, MODEL)


def smote(data):
    X, y = data.drop("target", axis=1), data["target"]
    sm = SMOTE(random_state=0)
    X, y = sm.fit_resample(X, y)
    data = pd.concat([X, y], axis=1)
    print("----- SMOTE -----")
    prediction.pred(data, MODEL)


if __name__ == "__main__":
    print("........................... D1 ............................")
    no_sampling(d1)
    downsampling(d1)
    upsampling(d1)
    up_and_dowsnsampling(d1, "d1")
    smote(d1)
    print("........................... D2 ............................")
    no_sampling(d2)
    downsampling(d2)
    upsampling(d2)
    up_and_dowsnsampling(d2, "d2")
    smote(d2)
    print("........................... D3 ............................")
    no_sampling(d3)
    downsampling(d3)
    upsampling(d3)
    up_and_dowsnsampling(d3, "d3")
    smote(d3)
    