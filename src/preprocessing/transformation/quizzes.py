#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transform quizzes variables."""

from datetime import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from cleaning import handle_outliers_quiz_time


OUTPUT_PATH = "../../results/visualization/"

DATE = datetime.today().strftime('%Y%m%d')


def remove_state_variables(dataset):
    for column in dataset.columns:
        if "state" in column:
            dataset.drop(column, axis=1, inplace=True)
    return dataset


def add_total_time(dataset):
    # handle time outliers
    for i in range(1, 5):
        if i == 4:
            type_ = "f"
        else:
            type_ = str(i)
        dataset = handle_outliers_quiz_time(dataset, type_)
    # add variable
    dataset["act_quiz_total_time"] = dataset["act_quiz1_time"] + \
        dataset["act_quiz2_time"] + dataset["act_quiz3_time"]
    return dataset


def add_avrg_grade(dataset):
    dataset["act_quiz_avrg_grade"] = (dataset["act_quiz1_grade"] +
                                      dataset["act_quiz2_grade"] +
                                      dataset["act_quiz3_grade"])/3
    return dataset


def remove_individual_quiz_time(dataset):
    variables = ["act_quiz1_time", "act_quiz2_time",
                 "act_quiz3_time", "act_quizf_time"]
    dataset.drop(variables, axis=1, inplace=True)
    return dataset


def add_scores(dataset):
    # scores (mean) for either retention or deduction performance
    # for dimensions visual/verbal/factual/practical
    dataset["score_r_visual"] = (dataset["act_quiz1_q1"] + dataset["act_quiz1_q3"] + dataset["act_quiz1_q4"])/3
    dataset["score_d_visual"] = (dataset["act_quiz1_q5"] + dataset["act_quiz1_q6"] + dataset["act_quiz1_q7"])/3
    dataset["score_r_verbal"] = (dataset["act_quiz1_q2"] + dataset["act_quiz2_q1"] + dataset["act_quiz2_q2"] + dataset["act_quiz2_q3"])/4
    dataset["score_d_verbal"] = (dataset["act_quiz2_q4"] + dataset["act_quiz2_q5"] + dataset["act_quiz2_q6"])/3
    dataset["score_r_factual"] = (dataset["act_quiz3_q1"] + dataset["act_quiz3_q2"])/2
    dataset["score_d_factual"] = (dataset["act_quiz3_q4"] + dataset["act_quiz3_q6"])/2
    dataset["score_r_practical"] = dataset["act_quiz3_q3"]
    dataset["score_d_practical"] = dataset["act_quiz3_q5"]
    # global scores (mean) for dimensions visual/verbal/factual/practical
    # 50-50 arbitrary balance for retention and deduction respective importance
    dataset["score_visual"] = (dataset["score_r_visual"] + dataset["score_d_visual"])/2
    dataset["score_verbal"] = (dataset["score_r_verbal"] + dataset["score_d_verbal"])/2
    dataset["score_factual"] = (dataset["score_r_factual"] + dataset["score_d_factual"])/2
    dataset["score_practical"] = (dataset["score_r_practical"] + dataset["score_d_practical"])/2
    # global scores (mean) for retention and deduction performance
    dataset["score_retention"] = (dataset["score_r_visual"] + dataset["score_r_verbal"] + dataset["score_r_factual"] + dataset["score_r_practical"])/4
    dataset["score_deduction"] = (dataset["score_d_visual"] + dataset["score_d_verbal"] + dataset["score_d_factual"] + dataset["score_d_practical"])/4
    # remove intermediate scores
    intermediate = ["score_r_visual", "score_d_visual", "score_r_verbal", "score_d_verbal",
                    "score_r_factual", "score_d_factual", "score_r_practical", "score_d_practical"]
    dataset.drop(columns=intermediate, inplace=True)
    return dataset


def remove_individual_and_quiz_grade(dataset):
    for column in dataset.columns:
        name = column.split("_")
        if ("grade" in name[-1]) and (column != "act_quizf_grade") and (column != "act_quiz_avrg_grade"):
            dataset.drop(column, axis=1, inplace=True)
        elif "q" in name[-1]:
            dataset.drop(column, axis=1, inplace=True)
    return dataset


def transformation(dataset):
    dataset = remove_state_variables(dataset)
    dataset = add_total_time(dataset)
    dataset = add_avrg_grade(dataset)
    dataset = remove_individual_quiz_time(dataset)
    dataset = add_scores(dataset)
    dataset = remove_individual_and_quiz_grade(dataset)
    dataset = dataset.rename(columns={"act_quizf_grade": "final_grade"})
    return dataset


def plot_transformation(dataset):
    # plot
    for column in dataset.columns:
        if "score" in column:
            ax = dataset[column].plot(kind="hist", color="b", rwidth=.96)
            ax.set_ylabel("Number of samples")
            ax.set_title(" ".join(column.split("_")))
            # set individual bar lables
            for i in ax.patches:
                ax.text(i.get_x()+.025, i.get_height()+.05,
                        str(int(i.get_height())),
                        fontsize=11, color='dimgrey', rotation=0)
            # save plot
            file_name = DATE + "_" + column + ".png"
            plt.savefig(OUTPUT_PATH + file_name)
            plt.show()
            print(f"Plot {file_name} saved in {OUTPUT_PATH}.", flush=True)
