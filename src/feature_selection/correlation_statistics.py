#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correlation statistics."""

from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_selection import (f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


OUTPUT_PATH = "../../results/feature_selection/"

DATE = datetime.today().strftime('%Y%m%d')


def dico(data):
    dico = dict()
    for i, feature_name in enumerate(data.columns):
        dico[i] = feature_name
    return dico


def save_plot(file_name):
    plt.savefig(OUTPUT_PATH + file_name, bbox_inches="tight")
    plt.show()


def plot_anova(scores):
    # plot
    ax = scores.plot(x="index", y=["score", "p-value"], kind="bar",
                     color=["r", "b"])
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("ANOVA F-value")
    ax.grid(axis='y')
    # save plot
    file_name = DATE + "_anova.png"
    save_plot(file_name)


def anova(data, target, dico=dico):
    print("ANOVA F-value computed.")
    scores = pd.DataFrame(f_classif(data, target))
    scores = scores.T.sort_values(by=0, ascending=False)
    scores.columns = ["score", "p-value"]
    scores["index"] = scores.index
    scores["index"] = scores["index"].apply(lambda x: dico[x])
    plot_anova(scores)


def plot_mi(score):
    # plot
    ax = score.plot(x="index", y=["mi"], kind="bar", color="b")
    ax.set_title("Estimated mutual information")
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.grid(axis='y')
    # save plot
    file_name = DATE + "_mutual_info.png"
    save_plot(file_name)


def mutual_info(data, target, dico=dico):
    rep = 100
    print(f"Mutual information computed over {rep} repetitions.")
    total_score = mutual_info_classif(data, target)
    for i in range(rep - 1):
        total_score += mutual_info_classif(data, target)
    total_score = total_score / rep
    score = pd.DataFrame(total_score, columns=["mi"]).sort_values(
        by="mi", ascending=False)
    score["index"] = score.index
    score["index"] = score["index"].apply(lambda x: dico[x])
    plot_mi(score)


def plot_kendall(score):
    # plot
    ax = score.plot(kind="bar", color="b")
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("Kendall's correlation coefficients")
    ax.grid(axis='y')
    # save plot
    file_name = DATE + "_kendall.png"
    save_plot(file_name)


def kendall(data, target):
    print("Kendall's coefficients computed.")
    score = pd.DataFrame(data).corrwith(target, method="kendall")
    score = score.sort_values(ascending=False, key=pd.Series.abs)
    plot_kendall(score)


def plot_logc_coeff(score):
    # plot
    ax = score.plot(kind="bar", color="b")
    ax.set_ylabel("Value")
    ax.set_xlabel("Feature")
    ax.set_title("Logistic classifier coefficients")
    ax.grid(axis='y')
    # save plot
    file_name = DATE + "_logc_coefficients.png"
    save_plot(file_name)


def logc_coefficients(data, target):
    print("Logistic classifier coefficients computed.")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(data, target)
    score = pd.DataFrame(np.transpose(model.coef_), data.columns,
                         columns=["coefficients"])
    score = score.sort_values("coefficients", ascending=False,
                              key=pd.Series.abs)
    plot_logc_coeff(score)
    

def scatter_matrix_plots(data, target):
    print("Scatter matrix computed.")
    variables1 = data.columns[:len(data.columns)//2]
    variables2 = data.columns[len(data.columns)//2:]
    df1 = pd.concat([data[variables1], target], axis=1)
    df2 = pd.concat([data[variables2], target], axis=1)
    # plot
    ax1 = sns.pairplot(df1, hue="target")
    # save plot
    file_name = DATE + "_scatter_matrix_1.png"
    save_plot(file_name)
    # plot
    ax2 = sns.pairplot(df2, hue="target")
    # save plot
    file_name = DATE + "_scatter_matrix_2.png"
    save_plot(file_name)


def correlation_matrix(data):
    print("Correlation matrix computed.")
    corr = pd.DataFrame(data).corr(method="kendall")
    # plot
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(corr, annot=True, xticklabels=corr.columns,
                yticklabels=corr.columns, cmap="Blues")
    # save plot
    file_name = DATE + "_correlation_matrix.png"
    save_plot(file_name)
