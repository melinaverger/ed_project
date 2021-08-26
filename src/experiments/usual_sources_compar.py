#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Usual sources comparison."""

import itertools
import prediction
import sys
sys.path.append('..')
import utils
import warnings
warnings.filterwarnings("ignore")


MODEL = "decision tree"


d1 = utils.load_dataset("../../data/05_modeling_dataset/")
d2 = utils.load_open("olc")
d2 = d2.dropna(axis=0, how="any")
d2.rename(columns={"final_result": "target"}, inplace=True)


demog_d1 = ["st_age", "st_ed_level", "st_ed_field"]
acad_d1 = ["act_quiz_avrg_grade"]
behav_d1 = ["%_completion", "st_nb_action", "act_quiz_total_time"]
dico_d1 = {"D": demog_d1, "A": acad_d1, "B": behav_d1}

demog_d2 = ["age_band", "highest_education"]
acad_d2 = ["avrg_past_grade"]
behav_d2 = ["total_sum_click"]
dico_d2 = {"D": demog_d2, "A": acad_d2, "B": behav_d2}


combi_usual = list()
a = ["D", "A", "B"]
for i in range(1, len(a)+1):
    elem = list(itertools.combinations(a, i))
    for c in elem:
        combi_usual.append(c)


def run(data, dico, combi):
    for i in range(len(combi)):
        sublen = len(combi[i])
        l = list()
        combi_name = ""
        for j in range(sublen):
            l = l + dico[combi[i][j]]
            combi_name = combi_name + combi[i][j] + " "
        print("----- {} -----".format(combi_name), flush=True)
        prediction.pred(data[l + ["target"]], MODEL)


if __name__ == "__main__":
    print("........................... D1 ............................")
    run(d1, dico_d1, combi_usual)
    print("........................... D2 ............................")
    run(d2, dico_d2, combi_usual)
