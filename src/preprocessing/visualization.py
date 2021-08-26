#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize the data."""

from cleaning import handle_outliers_quiz_time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils import load_dataset


DATASET_PATH = "../../data/03_dataset/"

OUTPUT_PATH = "../../results/visualization/"

DATE = datetime.today().strftime('%Y%m%d')


files_saved = list()


def save_plot(file_name):
    plt.savefig(OUTPUT_PATH + file_name)
    files_saved.append(file_name)
    print(f"Plot {file_name} saved in {OUTPUT_PATH}.", flush=True)


def plot_gender(data):
    y = data["st_gender"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks(x)
    ax.set_xticklabels(["1: Female", "2: Male", "3: NA"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Gender")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.35,
                i.get_height()+.1,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + "_gender.png"
    save_plot(file_name)
    plt.show()


def plot_age(data):
    y = data["st_age"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks(range(20, 65, 5))
    ax.set_ylabel("Number of samples")
    ax.set_title("Age")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x(),
                i.get_height()+.05,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + '_age.png'
    save_plot(file_name)
    plt.show()


def plot_language(data):
    y = data["st_french"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["1: French", "2: Other"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Language")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.35, i.get_height()+.25,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + "_language.png"
    save_plot(file_name)
    plt.show()


def plot_ed_level(data):
    y = data["st_ed_level"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks(list(range(1, 9)))
    ax.set_xticklabels(["+" + str(i) for i in range(1, 9)])
    ax.set_ylabel("Number of samples")
    ax.set_title("Level of education (from graduation)")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.3, i.get_height()+.15,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + "_ed_level.png"
    save_plot(file_name)
    plt.show()


def plot_ed_field(data):
    y = data["st_ed_field"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylabel("Number of samples")
    ax.set_title("Field of education")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.35, i.get_height()+.15,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    """ax.set_xticklabels([
    "1: Maths, Computer Science, Engineering, Technology...",
    "2: Health, Biology, Medicine...",
    "3: Humanities, Languages, Arts, Culture...",
    "4: Administration, Law, Management, Commerce..."],
    rotation=90)
    """
    # save plot
    file_name = DATE + "_ed_field.png"
    save_plot(file_name)
    plt.show()


def plot_motivation(data):
    y = data["st_motivation"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1: Interest", "2: Curiosity", "3: Participation"])
    ax.set_ylabel("Number of samples")
    ax.set_title("Initial motivation")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.35, i.get_height()+.2,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + "_motivation.png"
    save_plot(file_name)
    plt.show()


def plot_action(data):
    y = data["st_nb_action"].value_counts()
    x = list(y.index)
    # plot
    fig, ax = plt.subplots()
    ax.bar(x, y, color="b")
    ax.set_xticks(list(range(2, 22, 2)))
    ax.set_ylabel("Number of samples")
    ax.set_title("Total number of actions")
    # set individual bar lables
    for i in ax.patches:
        ax.text(i.get_x()+.15, i.get_height()+.05,
                str(round((i.get_height()), 2)),
                fontsize=11, color='dimgrey', rotation=0)
    # save plot
    file_name = DATE + "_action.png"
    save_plot(file_name)
    plt.show()


def plot_quiz_times(data):
    # handle outlier in quizf and quiz1
    data = handle_outliers_quiz_time(data, "f")
    data = handle_outliers_quiz_time(data, "1")
    # plot
    plt.hist(data["act_quiz1_time"], label='quiz1', rwidth=.96, alpha=1)
    plt.hist(data["act_quiz2_time"], label="quiz2", rwidth=.96, alpha=.8)
    plt.hist(data["act_quiz3_time"], label="quiz3", rwidth=.96, alpha=.6)
    plt.hist(data["act_quizf_time"], label="quizf", rwidth=.96, alpha=.6)
    plt.ylabel('Number of samples')
    plt.xlabel('Seconds')
    plt.xlim((-300, 1200))
    plt.title("Quiz times")
    plt.legend()
    # save plot
    file_name = DATE + "_quiz_times.png"
    save_plot(file_name)
    plt.show()


def plot_quiz_times_kde(data):
    # handle outlier in quizf and quiz1
    data = handle_outliers_quiz_time(data, "f")
    data = handle_outliers_quiz_time(data, "1")
    # plot
    fig, ax = plt.subplots()
    for i in range(1, 5):
        if i == 4:
            type_ = "f"
            name = "act_quiz" + type_ + "_time"
        else:
            type_ = str(i)
            name = "act_quiz" + type_ + "_time"
        data[name].plot(kind="kde", label='quiz' + type_)
    xmax = list()
    ymax = list()
    for i in range(4):
        x, y = ax.lines[i].get_xdata(), ax.lines[i].get_ydata()
        ymax_ = max(y)
        ypos_ = np.where(y == ymax_)
        xmax_ = x[ypos_]
        xmax.append(xmax_.item())
        ymax.append(ymax_)
    ax.vlines(xmax, 0, ymax, linestyle="dashed", color="black", alpha=.5)
    ax.scatter(xmax, ymax, color="black", alpha=.6)
    for i, txt in enumerate(xmax):
        ax.annotate(f"{int(txt)} s", xy=(xmax[i], ymax[i]),
                    xytext=(xmax[i]+100, ymax[i]-.0001))
    ax.set_xlim((-300, 1200))  # left, right = ax.set_xlim() to get limits
    ax.set_xlabel("Seconds")
    plt.title("Quiz times (kde)")
    plt.legend()
    # save plot
    file_name = DATE + "_quiz_times_kde.png"
    save_plot(file_name)
    plt.show()


def plot_quiz_grades(data):
    # plot
    plt.hist(data["act_quiz1_grade"], label="quiz1", rwidth=.96, alpha=.9)
    plt.hist(data["act_quiz2_grade"], label="quiz2", rwidth=.96, alpha=.8)
    plt.hist(data["act_quiz3_grade"], label="quiz3", rwidth=.96, alpha=.9)
    plt.hist(data["act_quizf_grade"], label="quizf", rwidth=.96, alpha=.8)
    plt.ylabel('Number of samples')
    plt.title('Quiz grades')
    plt.xlim((0, 10))
    plt.legend()
    # save plot
    file_name = DATE + "_quiz_grades.png"
    save_plot(file_name)
    plt.show()


def plot_quiz_grades_kde(data):
    # plot
    fig, ax = plt.subplots()
    for i in range(1, 5):
        if i == 4:
            type_ = "f"
            name = "act_quiz" + type_ + "_grade"
        else:
            type_ = str(i)
            name = "act_quiz" + type_ + "_grade"
        data[name].plot(kind="kde", label='quiz' + type_)
    xmax = list()
    ymax = list()
    for i in range(4):
        x, y = ax.lines[i].get_xdata(), ax.lines[i].get_ydata()
        ymax_ = max(y)
        ypos_ = np.where(y == ymax_)
        xmax_ = x[ypos_]
        xmax.append(xmax_.item())
        ymax.append(ymax_)
    ax.vlines(xmax, 0, ymax, linestyle="dashed", color="black", alpha=.5)
    ax.scatter(xmax, ymax, color="black", alpha=.6)
    for i, txt in enumerate(xmax):
        ax.annotate(round(txt, 2), xy=(xmax[i], ymax[i]),
                    xytext=(xmax[i] + .2, ymax[i]))
    ax.set_xlim((0, 10))
    plt.title("Quiz grades (kde)")
    plt.legend()
    # save plot
    file_name = DATE + "_quiz_grades_kde.png"
    save_plot(file_name)
    plt.show()


def main():
    print("********************** Visualization **********************", flush=True)
    data = load_dataset()
    count_nan_rows = data.shape[0] - data.dropna(axis=0, how="any",
                                                 inplace=False).shape[0]
    print(f"The dataset contains {data.shape[0]} samples and {data.shape[1]} "
          f"columns (including {count_nan_rows} incomplete samples).",
          flush=True)
    plot_gender(data)
    plot_age(data)
    plot_language(data)
    plot_ed_level(data)
    plot_ed_field(data)
    plot_motivation(data)
    plot_action(data)
    plot_quiz_times(data)
    plot_quiz_times_kde(data)
    plot_quiz_grades(data)
    plot_quiz_grades_kde(data)


if __name__ == "__main__":
    main()
