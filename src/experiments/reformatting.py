#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reformat datasets to keep common features."""

import pandas as pd
import zipfile
import sys
sys.path.append('..')
import utils


DATA_PATH = "../../data/open_olc/anonymised_data/"
INTERM_PATH = "../../data/open_olc/saved_data/"
OUTPUT_PATH = "../../data/06_common_data/"


#############################
# OLC dataset preprocessing #
#############################


def studentAssessment_prepro():
    # load data
    zf = zipfile.ZipFile(DATA_PATH + "studentAssessment.csv.zip") 
    data_ass = pd.read_csv(zf.open("studentAssessment.csv"))
    zf = zipfile.ZipFile(DATA_PATH + "assessments.csv.zip") 
    data_descr = pd.read_csv(zf.open("assessments.csv"))

    # merge
    data = data_ass.merge(data_descr, how="left", on=["id_assessment"])

    # transformations
    # 1. make the average of the training grades
    # 2. set an "exam_grade" column (None or score)
    # 3. set a "completion" column (1 or 0)
    # 4. set a "weigthed" column (1 or 0) for the training assessments
    l_id = list()
    l_module = list()
    l_pres = list()

    l_avrg_past_grade = list()
    l_exam_grade = list()
    l_completed = list()
    l_weighted = list()

    for idd in data["id_student"].unique():
        subdf = data[data["id_student"] == idd]
        for code_mod in subdf["code_module"].unique():
            subdf2 = subdf[subdf["code_module"] == code_mod]
            for code_pres in subdf2["code_presentation"].unique():
                new = subdf2[subdf2["code_presentation"] == code_pres]

                l_module.append(code_mod)
                l_id.append(idd)
                l_pres.append(code_pres)

                # compute the average + take the exam grade if exists
                s = 0
                w = 0
                exam_grade = None
                for i in range(new.shape[0]):
                    if new["assessment_type"].iloc[i] != "Exam":
                        s += new["score"].iloc[i]*new["weight"].iloc[i]
                        w += new["weight"].iloc[i]
                    else:
                        exam_grade = new["score"].iloc[i]
                l_exam_grade.append(exam_grade)

                # "weighted" column
                if w == 0:
                    l_weighted.append(0)
                    l_avrg_past_grade.append(new["score"].mean())
                else:
                    l_weighted.append(1)
                    l_avrg_past_grade.append(s/w)

                # "completion" column
                if w == 100:
                    l_completed.append(1)
                else:
                    l_completed.append(0)

    dfnew = pd.DataFrame(list(zip(l_module, l_pres, l_id, l_avrg_past_grade,
                                  l_exam_grade, l_completed, l_weighted)),
                         columns=["code_module", "code_presentation",
                                  "id_student", "avrg_past_grade",
                                  "exam_grade", "completed", "weighted"])

    dfnew.to_csv(INTERM_PATH + "data_assessments_prepro.csv", index=False)


def studentInfo_prepro():
    # load data
    zf = zipfile.ZipFile(DATA_PATH + "studentInfo.csv.zip") 
    data = pd.read_csv(zf.open("studentInfo.csv"))
    data = data[data["final_result"] != "Withdrawn"]
    data = data[data["final_result"] != "Distinction"]

    drop_cols = ["region", "imd_band", "num_of_prev_attempts",
                 "studied_credits", "disability"]
    data = data.drop(drop_cols, axis=1)

    # format gender
    data["gender"] = data["gender"].replace(to_replace=["M", "F"],
                                            value=[2, 1])

    # format level of education
    data = data[data["highest_education"] != "Lower Than A Level"]
    data = data[data["highest_education"] != "No Formal quals"]
    data["highest_education"] = data["highest_education"].replace(to_replace=["A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"],
                                                                  value=[1, 2, 3])
    data["highest_education"] = data["highest_education"].astype(int)

    # format age
    data["age_band"] = data["age_band"].replace(to_replace=["0-35", "35-55", "55<="],
                                                value=[1, 2, 3])

    # format final_result
    data["final_result"] = data["final_result"].replace(to_replace=["Pass", "Fail"],
                                                        value=[1, 0])
    data["final_result"] = data["final_result"].astype(int)

    data.to_csv(INTERM_PATH + "data_info_prepro.csv", index=False)


def studentVle_prepro():
    # load data
    zf = zipfile.ZipFile(DATA_PATH + "studentVle.csv.zip") 
    data = pd.read_csv(zf.open("studentVle.csv"))

    drop_cols = ["id_site", "date"]
    data = data.drop(drop_cols, axis=1)

    # gathering per triplet (student, module, presentation)
    l_id = list()
    l_click = list()
    l_module = list()
    l_pres = list()

    for idd in data["id_student"].unique():
        subdf = data[data["id_student"] == idd]
        for code_pres in subdf["code_presentation"].unique():
            subdf2 = subdf[subdf["code_presentation"] == code_pres]
            for code_mod in subdf2["code_module"].unique():
                new = subdf2[subdf2["code_module"] == code_mod]
                l_click.append(new["sum_click"].sum())
                l_module.append(code_mod)
                l_id.append(idd)
                l_pres.append(code_pres)

    dfnew = pd.DataFrame(list(zip(l_module, l_pres, l_id, l_click)),
                         columns=["code_module", "code_presentation",
                                  "id_student", "total_sum_click"])

    dfnew.to_csv(INTERM_PATH + "data_Vle_prepro.csv", index=False)


def student_merge_dataset():
    info = pd.read_csv("../../data/open_olc/saved_data/data_info_prepro.csv")
    vle = pd.read_csv("../../data/open_olc/saved_data/data_Vle_prepro.csv")
    grd = pd.read_csv("../../data/open_olc/saved_data/data_assessments_prepro.csv")

    # inner merge
    data = info.merge(vle, how="inner", on=["code_module", "code_presentation", "id_student"])
    data = data.merge(grd, how="inner", on=["code_module", "code_presentation", "id_student"])

    data.to_csv(INTERM_PATH + "data_merged.csv", index=False)


def preprocess_olc_data():
    studentAssessment_prepro()
    studentInfo_prepro()
    studentVle_prepro()
    student_merge_dataset()


################
# Reformatting #
################


def age_band_d1(age_int):
    if age_int < 35:
        return 1
    elif 35 <= age_int < 55:
        return 2
    else:
        return 3


def age_band_d3(age_str):
    if age_str == "{19-34}":
        return 1
    elif age_str == "{34-54}":
        return 2
    else:
        return 3


def highest_education_d1(ed_level_int):
    if ed_level_int > 5:
        return 3
    elif 1 < ed_level_int <= 5:
        return 2
    else:
        return 1


def highest_education_d3(ed_str):
    if ed_str == "Master's Degree (or equivalent)":
        return 2
    elif ed_str == "Completed 4-year college degree":
        return 2
    elif ed_str == "Some college, but have not finished a degree":
        return 1
    elif ed_str == "Some graduate school":
        return 3
    elif ed_str == "Ph.D., J.D., or M.D. (or equivalent)":
        return 3
    elif ed_str == "Completed 2-year college degree":
        return 2
    elif ed_str == "High School or College Preparatory School":
        return 1


def target_class_d3(grd):
    if float(grd) >= 0.5:
        return 1
    else:
        return 0


def common_features():
    d1 = utils.load_dataset("../../data/05_modeling_dataset/")
    d2 = utils.load_open("olc")
    d3 = utils.load_open("cnp")

    kept_cols_1 = ["st_age", "st_ed_level", "st_nb_action", "target"]
    kept_cols_2 = ["age_band", "highest_education", "total_sum_click", "final_result"]
    kept_cols_3 = ["course_id_DI", "userid_DI", "age_DI", "LoE_DI", "nevents", "grade"]
    d1 = d1[kept_cols_1]
    d2 = d2[kept_cols_2]
    d3 = d3[kept_cols_3]

    d1["st_age"] = d1["st_age"].apply(age_band_d1)
    d1["st_ed_level"] = d1["st_ed_level"].apply(highest_education_d1)

    d2.rename(columns={"final_result": "target"}, inplace=True)
    
    d3 = d3.drop_duplicates(subset=['course_id_DI', 'userid_DI'], keep='first')
    d3 = d3.drop(["course_id_DI", "userid_DI"], axis=1)
    d3 = d3.dropna(axis=0, how="any")
    d3 = d3[d3["age_DI"] != "{}"]
    d3["age_DI"] = d3["age_DI"].apply(age_band_d3)
    d3 = d3[d3["LoE_DI"] != "Missing"]
    d3 = d3[d3["LoE_DI"] != "None of these"]
    d3["LoE_DI"] = d3["LoE_DI"].apply(highest_education_d3)
    d3["grade"] = d3["grade"].apply(target_class_d3)
    d3.rename(columns={"grade": "target"}, inplace=True)

    d1.to_csv(OUTPUT_PATH + "d1.csv", index=False)
    d2.to_csv(OUTPUT_PATH + "d2.csv", index=False)
    d3.to_csv(OUTPUT_PATH + "d3.csv", index=False)


if __name__ == "__main__":
    print("Preprocessing OLC open dataset...", flush=True)
    preprocess_olc_data()
    print("OLC dataset has been preprocessed and saved in "
          "{}.".format(INTERM_PATH))
    common_features()
    print("D1: collected data. D2: OLC dataset. D3: CNP dataset.")
    print("Common features of D1, D2 and D3 have been reformatted and "
          "new datasets saved in {}.".format(OUTPUT_PATH))
