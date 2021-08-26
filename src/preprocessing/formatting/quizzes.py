#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the quizzes data to a more easily manipulated format,
without changing them."""

import locale
import datetime
import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
import utils


DATA_PATH = "../../../data/01_raw_data/"

OUTPUT_PATH = "../../../data/02_converted_data/"


def rename_columns(data, type_):
    if type_ not in ["1", "2", "3", "f"]:
        raise ValueError("rename_columns() takes only '1', '2', '3' and 'f' as type_ argument.")
    data.rename(columns={"ID": "st_id",
                         "État": "act_quiz" + type_ + "_state",
                         "Commencé le": "act_quiz" + type_ + "_start_date",
                         "Terminé": "act_quiz" + type_ + "_end_date",
                         "Temps utilisé": "act_quiz" + type_ + "_time",
                         "Note/10,00": "act_quiz" + type_ + "_grade"},
                inplace=True)
    return data


def convert_state_values(data):
    data.replace(to_replace=["Terminé", "En cours"],
                 value=[1, 0],
                 inplace=True)
    return data


def date_format(date_string):
    locale.setlocale(locale.LC_TIME, "fr_FR")  # french
    format_ = "%d %B %Y  %H:%M"
    if type(date_string) == str:
        if date_string != "-":
            date_time = datetime.datetime.strptime(date_string, format_)
        else:
            date_time = None
        return date_time
    else:
        return date_string


def convert_date(data):
    for column in data.columns:
        if "date" in column:
            data[column] = data[column].apply(date_format)
    return data


def replace_date_to_attempt(df, type_):
    # name of the studied column
    column_name = "act_quiz" + type_ + "_start_date"
    for st_id in df.st_id.unique():
        # keep particular student data only
        subdf = df.loc[df["st_id"] == st_id]
        index_subdf = subdf.index.values
        # if several attempts
        if len(subdf[column_name]) > 1:
            # remember dates
            date_times = list()
            for i in range(len(subdf[column_name])):
                time_stamp = subdf[column_name].iloc[i]
                date_times.append(time_stamp)
            # compare dates
            index_by_date = np.argsort(date_times)
            # add a correspondant number of attempt
            for count, ind in enumerate(index_subdf):
                df.loc[ind, "act_quiz" + type_ + "_nattempt"] = index_by_date[count] + 1
        # if only one attempt
        else:
            if subdf["act_quiz" + type_ + "_grade"].isnull().values.any():
                df.loc[int(index_subdf), "act_quiz" + type_ + "_nattempt"] = None
            else:
                df.loc[int(index_subdf), "act_quiz" + type_ + "_nattempt"] = 1
    return df


def time_format(time_string):
    format_short = "%S s"
    format_med1 = "%M min"
    format_med2 = "%M min %S s"
    format_long11 = "%H heure"
    format_long12 = "%H heures"
    format_long21 = "%H heure %S s"
    format_long22 = "%H heures %S s"
    format_long31 = "%H heure %M min"
    format_long32 = "%H heures %M min"
    format_long41 = "%H heure %M min %S s"
    format_long42 = "%H heures %M min %S s"
    if type(time_string) == str:
        if time_string != "-":
            if ("heure" in time_string) and ("min" in time_string):
                splitted = time_string.split(" ")
                # plurial hours
                if int(splitted[0]) > 1:
                    # if seconds
                    if splitted[-1] == "s":
                        time = datetime.datetime.strptime(time_string, format_long42)
                    else:
                        time = datetime.datetime.strptime(time_string, format_long32)
                else:
                    # if seconds
                    if splitted[-1] == "s":
                        time = datetime.datetime.strptime(time_string, format_long41)
                    else:
                        time = datetime.datetime.strptime(time_string, format_long31)
            elif "heure" in time_string:
                splitted = time_string.split(" ")
                # plurial hours
                if int(splitted[0]) > 1:
                    # if seconds
                    if splitted[-1] == "s":
                        time = datetime.datetime.strptime(time_string, format_long22)
                    else:
                        time = datetime.datetime.strptime(time_string, format_long12)
                else:
                    # if seconds
                    if splitted[-1] == "s":
                        time = datetime.datetime.strptime(time_string, format_long21)
                    else:
                        time = datetime.datetime.strptime(time_string, format_long11)
            elif "min" in time_string:
                splitted = time_string.split(" ")
                # if seconds
                if splitted[-1] == "s":
                    time = datetime.datetime.strptime(time_string, format_med2)
                else:
                    time = datetime.datetime.strptime(time_string, format_med1)
            elif "s" in time_string:
                time = datetime.datetime.strptime(time_string, format_short)
            else:
                raise ValueError("Unknown time format.")
            # convert in seconds
            timedelta = time - datetime.datetime(1900, 1, 1)
            seconds = timedelta.total_seconds()
        else:
            seconds = None
        return seconds
    else:
        return time_string


def convert_time(data):
    for column in data.columns:
        if "time" in column:
            data[column] = data[column].apply(time_format)
    return data


def grade_format(grade_string):
    if type(grade_string) == str:
        if grade_string != "-":
            grade = float(grade_string.replace(",", "."))
        else:
            grade = None
        return grade
    else:
        return grade_string


def convert_grade(data):
    for column in data.columns:
        if "grade" in column:
            data[column] = data[column].apply(grade_format)
    return data


def convert_questions(data, type_):
    count = 1
    for column in data.columns:
        if "Q." in column:
            data[column].replace(to_replace="-", value="0,00", inplace=True)
            # replace comma by point and convert str to float
            data[column] = [float(qgrade.replace(",", ".")) if type(qgrade) == str else qgrade for qgrade in data[column]]
            # split column name to keep the maximal grade information
            max_grade = float(column.split("/")[-1].replace(",", "."))
            # modify column values
            data[column] = data[column] / max_grade
            # rename column
            data.rename(columns={column: "act_quiz" + type_ + "_q" + str(count)}, inplace=True)
            count += 1
    return data


def preprocess_data():
    names = ["quiz1", "quiz2", "quiz3", "quizf"]
    for i, name in enumerate(names, 1):
        quiz_name = utils.get_file_or_error(DATA_PATH, name)
        data = pd.read_csv(DATA_PATH + quiz_name)
        if i == 4:
            type_ = "f"
        else:
            type_ = str(i)
        data = convert_state_values(data)
        data = convert_questions(data, type_)
        data = rename_columns(data, type_)
        data = convert_grade(data)
        data = convert_time(data)
        data = convert_date(data)
        data = replace_date_to_attempt(data, type_)
        data.drop("act_quiz" + type_ + "_end_date", axis=1, inplace=True)  # redundant with activity data
        data = data[data["act_quiz" + type_ + "_nattempt"] == 1]  # keep only the 1st attempt
        data.drop("act_quiz" + type_ + "_nattempt", axis=1, inplace=True)
        utils.save_or_replace(data, name, OUTPUT_PATH)


if __name__ == "__main__":
    preprocess_data()
    print("Quizzes data have been converted and saved in "
          "{}.".format(OUTPUT_PATH))
