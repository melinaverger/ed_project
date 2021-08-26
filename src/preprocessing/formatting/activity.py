#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the activity data to a more easily manipulated format,
without changing them."""

import locale
import datetime
import pandas as pd
import sys
sys.path.append('../..')
import utils


DATA_PATH = "../../../data/01_raw_data/"

OUTPUT_PATH = "../../../data/02_converted_data/"


def rename_columns(data):
    data.rename(columns={"ID": "st_id",
                         "Questionnaire préalable": "act_questionnaire",
                         "Questionnaire préalable - Date d'achèvement": "act_questionnaire_end_date",
                         "Les aires fonctionnelles du cerveau": "res_image_analysis",
                         "Les aires fonctionnelles du cerveau - Date d'achèvement": "res_image_analysis_end_date",
                         "Auto-évaluation 1": "act_quiz1",
                         "Auto-évaluation 1 - Date d'achèvement": "act_quiz1_end_date",
                         "A l'extérieur et à l'intérieur du cerveau": "res_text_explanations",
                         "A l'extérieur et à l'intérieur du cerveau - Date d'achèvement": "res_text_explanations_end_date",
                         "Auto-évaluation 2": "act_quiz2",
                         "Auto-évaluation 2 - Date d'achèvement": "act_quiz2_end_date",
                         "Caractéristiques": "res_factual",
                         "Caractéristiques - Date d'achèvement": "res_factual_end_date",
                         "Informations pratiques": "res_practical",
                         "Informations pratiques - Date d'achèvement": "res_practical_end_date",
                         "Auto-évaluation 3": "act_quiz3",
                         "Auto-évaluation 3 - Date d'achèvement": "act_quiz3_end_date",
                         "Quiz final": "act_quizf",
                         "Quiz final - Date d'achèvement": "act_quizf_end_date",
                         "Cours terminé": "course_end_date"},
                inplace=True)
    return data


def convert_state_values(data):
    data.replace(to_replace=["Terminé", "Pas terminé"],
                 value=[1, 0],
                 inplace=True)
    return data


def date_format(date_string):
    locale.setlocale(locale.LC_TIME, "fr_FR")  # french
    format_ = "%d %B %y, %H:%M"
    abrevation_dict = {"juil.": "juillet"}
    if type(date_string) == str:
        for abrevation in abrevation_dict.keys():
            if abrevation in date_string:
                date_string = date_string.replace(abrevation,
                                                  abrevation_dict[abrevation])
        date_time = datetime.datetime.strptime(date_string, format_)
        return date_time
    else:
        return date_string


def convert_date(data):
    for column in data.columns:
        if "date" in column:
            data[column] = data[column].apply(date_format)
    return data


def preprocess_data():
    activity_name = utils.get_file_or_error(DATA_PATH, "activity")
    data = pd.read_csv(DATA_PATH + activity_name)
    data = rename_columns(data)
    data = convert_state_values(data)
    data = convert_date(data)
    utils.save_or_replace(data, "activity", OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_data()
    print("Activity data have been converted and saved in "
          "{}.".format(OUTPUT_PATH))
