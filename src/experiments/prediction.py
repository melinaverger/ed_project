#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Models and evaluation."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


def display_results(scores):
    print("Accuracy: \t {} (+/- {})".format(round(scores["test_accuracy"].mean()*100, 2),
                                            round(scores["test_accuracy"].std()*100, 2)))
    print("Precision: \t {} (+/- {})".format(round(scores["test_precision_macro"].mean()*100, 2),
                                             round(scores["test_precision_macro"].std()*100, 2)))
    print("Recall: \t \t {} (+/- {})".format(round(scores["test_recall_macro"].mean()*100, 2),
                                          round(scores["test_recall_macro"].std()*100, 2)))
    print("F1-score: \t {} (+/- {})".format(round(scores["test_f1_macro"].mean()*100, 2),
                                            round(scores["test_f1_macro"].std()*100, 2)))


def dt_model(X, y):
    # parameters
    criterion = ["gini"]
    max_depth = [1, 2, 3, 4, 5, 6]
    tuned_params = {"criterion": criterion, "max_depth": max_depth}
    # model
    model_dt = DecisionTreeClassifier(random_state=0)
    # grid search
    print("Grid search...")
    gs = GridSearchCV(estimator=model_dt,
                  param_grid=tuned_params,
                  scoring="accuracy",
                  cv=10,
                  verbose=0)
    gs.fit(X, y)
    new_model = gs.best_estimator_
    print(new_model)
    print("Fit model...")
    new_model.fit(X, y)
    f_imp = [round(x, 2) for x in list(new_model.feature_importances_)]
    print("Feature importance: ", f_imp)
    # evaluation
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(new_model, X, y, cv=10, scoring=scoring)
    return scores


def pred(data, model):
    X, y = data.drop("target", axis=1), data["target"]
    if model == "decision tree":
        res = dt_model(X, y)
        display_results(res)