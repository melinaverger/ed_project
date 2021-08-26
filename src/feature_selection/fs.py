#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature selection."""

from normalization import normalize
from low_variance import low_variance_removal
import correlation_statistics
import wrappers
import pandas as pd
import sys
sys.path.append('..')
from utils import get_file_or_error


DATA_PATH = "../../data/05_modeling_dataset/"

OUTPUT_PATH = "../../results/feature_selection/"


def main():
    file_name = get_file_or_error(DATA_PATH, "modeling_dataset")
    dataset = pd.read_csv(DATA_PATH + file_name)
    X, y = dataset.drop("target", axis=1), dataset["target"]
    print("********************** Normalization **********************", flush=True)
    X = normalize(X)
    print("Features normalized.", flush=True)
    print("************** Low-variance features removal **************", flush=True)
    X = low_variance_removal(X)
    dico = correlation_statistics.dico(X)
    print("****************** Correlation statistics *****************", flush=True)
    correlation_statistics.anova(X, y, dico)
    correlation_statistics.mutual_info(X, y, dico)
    correlation_statistics.kendall(X, y)
    correlation_statistics.logc_coefficients(X, y)
    correlation_statistics.scatter_matrix_plots(X, y)
    correlation_statistics.correlation_matrix(X)
    print(f"Variable ranking results saved in {OUTPUT_PATH}.", flush=True)
    print("************************ Wrappers *************************", flush=True)
    n_optimal = wrappers.rfecv(X, y)
    wrappers.forward_elimination(X, y, n_optimal)
    wrappers.backward_elimination(X, y, n_optimal)
    wrappers.rfe(X, y, n_optimal)


if __name__ == "__main__":
    main()
