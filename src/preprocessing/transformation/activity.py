#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transform activity variables."""

from datetime import datetime
import matplotlib.pyplot as plt


OUTPUT_PATH = "../../results/visualization/"

DATE = datetime.today().strftime('%Y%m%d')


def remove_date_variables(dataset):
    for column in dataset.columns:
        if "date" in column:
            dataset.drop(column, axis=1, inplace=True)
    return dataset


def transformation(dataset):
    # add %_completion variable
    variables = ["act_questionnaire", "res_image_analysis", "act_quiz1",
                 "res_text_explanations", "act_quiz2", "res_factual",
                 "res_practical", "act_quiz3", "act_quizf"]
    dataset["%_completion"] = dataset[variables].sum(axis=1)/len(variables)
    dataset.drop(variables, axis=1, inplace=True)
    # remove date variables (not activity data only)
    dataset = remove_date_variables(dataset)
    return dataset


def plot_transformation(dataset):
    # plot
    ax = dataset["%_completion"].plot(kind="hist", color="b",
                                      bins=10, rwidth=.96)
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("%")
    ax.set_title("Percentage of completion")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.03,
                i.get_height()+.2,
                str(int((i.get_height()))),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + "_completion.png"
    plt.savefig(OUTPUT_PATH + file_name)
    plt.show()
    print(f"Plot {file_name} saved in {OUTPUT_PATH}.", flush=True)
