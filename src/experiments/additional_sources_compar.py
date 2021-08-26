#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional sources comparison."""

import itertools
from usual_sources_compar import run
import sys
sys.path.append('..')
import utils


d1 = utils.load_dataset("../../data/05_modeling_dataset/")


demog_d1 = ["st_age", "st_ed_level", "st_ed_field"]
acad_d1 = ["act_quiz_avrg_grade"]
behav_d1 = ["%_completion", "st_nb_action", "act_quiz_total_time"]
personality_d1 = ["st_motivation"]
lp_d1 = ["score_visual", "score_verbal", "score_factual", "score_practical",
         "score_retention", "score_deduction"]
dico_d1 = {"D": demog_d1, "A": acad_d1, "B": behav_d1,
           "P": personality_d1, "L": lp_d1}


combi_additional = list()
a = ["D", "A", "B", "P", "L"]
for i in range(1, len(a)+1):
    elem = list(itertools.combinations(a, i))
    for c in elem:
        combi_additional.append(c)


if __name__ == "__main__":
    print("........................... D1 ............................")
    run(d1, dico_d1, combi_additional)
