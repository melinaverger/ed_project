#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature preprocessing."""

from transformation import activity, questionnaire, quizzes
from cleaning import cleaning
from target import target
import sys
sys.path.append('..')
from utils import load_dataset, save_or_replace


OUTPUT_PATH = "../../data/04_transformed_dataset/"

OUTPUT_PATH_2 = "../../data/05_modeling_dataset/"


TARGET_TYPE = "binary classification"


def main():
    dataset = load_dataset()
    print("********************* Transformation **********************", flush=True)
    print("The dataset has been transformed.", flush=True)
    print("New data visualization is available.", flush=True)
    dataset = activity.transformation(dataset)
    activity.plot_transformation(dataset)
    dataset = questionnaire.transformation(dataset)
    dataset = quizzes.transformation(dataset)
    quizzes.plot_transformation(dataset)
    print("************************ Cleaning *************************", flush=True)
    dataset = cleaning(dataset)
    save_or_replace(dataset, "transformed_dataset", OUTPUT_PATH)
    print(f"The dataset has been cleand and saved in {OUTPUT_PATH}.", flush=True)
    print(f"Text variables have been separated and saved in {OUTPUT_PATH}.", flush=True)
    print("********************* Target variable *********************", flush=True)
    dataset = target(dataset, TARGET_TYPE)
    save_or_replace(dataset, "modeling_dataset", OUTPUT_PATH_2)
    print(f"New dataset saved in {OUTPUT_PATH_2}.", flush=True)


if __name__ == "__main__":
    main()
