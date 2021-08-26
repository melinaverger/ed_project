#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learning algorithms comparison and transfer learning."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


DATA_PATH = "../../data/06_common_data/"


d1_unbalanced = pd.read_csv(DATA_PATH + "d1.csv")
d2_unbalanced = pd.read_csv(DATA_PATH + "d2.csv")
d3_unbalanced = pd.read_csv(DATA_PATH + "d3.csv")

d1_balanced = pd.read_csv(DATA_PATH + "d1_balanced.csv")
d2_balanced = pd.read_csv(DATA_PATH + "d2_balanced.csv")
d3_balanced = pd.read_csv(DATA_PATH + "d3_balanced.csv")


def normalize(x):
    feature_names = x.columns
    scaler = MinMaxScaler()
    x = pd.DataFrame(scaler.fit_transform(x))
    x.columns = feature_names
    return x


# normalize data
X1_train, y1_train = d1_balanced.drop("target", axis=1), d1_balanced["target"]
X1_train, y1_train = normalize(X1_train.copy()), y1_train.copy()

X1_test, y1_test = d1_unbalanced.drop("target", axis=1), d1_unbalanced["target"]
X1_test, y1_test = normalize(X1_test.copy()), y1_test.copy()


X2_train, y2_train = d2_balanced.drop("target", axis=1), d2_balanced["target"]
X2_train, y2_train = normalize(X2_train.copy()), y2_train.copy()

X2_test, y2_test = d2_unbalanced.drop("target", axis=1), d2_unbalanced["target"]
X2_test, y2_test = normalize(X2_test.copy()), y2_test.copy()


X3_train, y3_train = d3_balanced.drop("target", axis=1), d3_balanced["target"]
X3_train, y3_train = normalize(X3_train.copy()), y3_train.copy()

X3_test, y3_test = d3_unbalanced.drop("target", axis=1), d3_unbalanced["target"]
X3_test, y3_test = normalize(X3_test.copy()), y3_test.copy()


# class-imbalance ratios
print("Training class-imbalance ratios:")
print("D1: ", round(y1_train.mean()*100, 2))
print("D2: ", round(y2_train.mean()*100, 2))
print("D3: ", round(y3_train.mean()*100, 2))

print("Test class-imbalance ratios:")
print("D1: ", round(y1_test.mean()*100, 2))
print("D2: ", round(y2_test.mean()*100, 2))
print("D3: ", round(y3_test.mean()*100, 2))


def truncate(num,n):
    temp = str(num)
    for x in range(len(temp)):
        if temp[x] == '.':
            try:
                return float(temp[:x+n+1])
            except:
                return float(temp)      
    return float(temp)


def get_results(MODEL, X_TRAIN, Y_TRAIN):
    # best model training
    MODEL.fit(X_TRAIN, Y_TRAIN)
    
    # predictions
    y1_pred = MODEL.predict(X1_test)
    y2_pred = MODEL.predict(X2_test)
    y3_pred = MODEL.predict(X3_test)

    l_prec = [precision_score(y1_test, y1_pred, average='macro')*100, precision_score(y2_test, y2_pred, average='macro')*100, precision_score(y3_test, y3_pred, average='macro')*100]
    l_rec = [recall_score(y1_test, y1_pred, average='macro')*100, recall_score(y2_test, y2_pred, average='macro')*100, recall_score(y3_test, y3_pred, average='macro')*100]
    l_f1 = [f1_score(y1_test, y1_pred, average='macro')*100, f1_score(y2_test, y2_pred, average='macro')*100, f1_score(y3_test, y3_pred, average='macro')*100]
    l_acc = [accuracy_score(y1_test, y1_pred)*100, accuracy_score(y2_test, y2_pred)*100, accuracy_score(y3_test, y3_pred)*100]

    l_prec_round = [truncate(val, 2) for val in l_prec]
    l_rec_round = [truncate(val, 2) for val in l_rec]
    l_f1_round = [truncate(val, 2) for val in l_f1]
    l_acc_round = [truncate(val, 2) for val in l_acc]

    results = pd.DataFrame(list(zip(l_acc_round, l_prec_round, l_rec_round, l_f1_round)),
                           index=['D1', 'D2', 'D3'],
                           columns=['acc', 'preci', 'recall', 'f1'])
    
    return results


def DT_model(X_TRAIN, Y_TRAIN):
    # parameters
    criterion = ["gini"]
    max_depth = [1, 2, 3, 4, 5, 6]
    tuned_params = {'criterion': criterion, "max_depth": max_depth}

    # model
    model_dt = DecisionTreeClassifier(random_state=0)

    # grid search
    gs = GridSearchCV(estimator=model_dt,
                      param_grid=tuned_params,
                      scoring="accuracy",
                      cv=10,
                      verbose=0)
    
    gs.fit(X_TRAIN, Y_TRAIN)

    new_model = gs.best_estimator_
    
    return new_model


def SVC_model(X_TRAIN, Y_TRAIN):
    # parameters
    tuned_params = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001, 'scale', 'auto'], 'kernel':['linear','rbf']}

    # model
    model_svc = SVC(random_state=0)

    # grid search
    gs = GridSearchCV(estimator=model_svc,
                      param_grid=tuned_params,
                      scoring="accuracy",
                      cv=10,
                      verbose=0) 

    gs.fit(X_TRAIN, Y_TRAIN)

    new_model = gs.best_estimator_
    
    return new_model


def RF_model(X_TRAIN, Y_TRAIN):
    # parameters
    tuned_params = {'n_estimators': [50, 100, 200, 500],
                  'max_features': ['auto'],
                  'max_depth' : [1, 2, 3, 4, 5, 6],
                  'criterion' :['gini']}
    
    # model
    model_rf = RandomForestClassifier(random_state=0)

    # grid search
    gs = GridSearchCV(estimator=model_rf,
                      param_grid=tuned_params,
                      scoring="accuracy",
                      cv=10,
                      verbose=0) 

    gs.fit(X_TRAIN, Y_TRAIN)

    new_model = gs.best_estimator_
    
    return new_model


if __name__ == "__main__":
    print("..................... Decision trees ......................", flush=True)
    print('----- Training on D1, testing on D1, D2 and D3')
    model1 = DT_model(X1_train, y1_train)
    print(get_results(model1, X1_train, y1_train))
    print('----- Training on D2, testing on D1, D2 and D3')
    model2 = DT_model(X2_train, y2_train)
    print(get_results(model2, X2_train, y2_train))
    print('----- Training on D3, testing on D1, D2 and D3')
    model3 = DT_model(X3_train, y3_train)
    print(get_results(model3, X3_train, y3_train))
    print(".......................... SVM ............................", flush=True)
    print('----- Training on D1, testing on D1, D2 and D3')
    model1 = SVC_model(X1_train, y1_train)
    print(get_results(model1, X1_train, y1_train))
    print('----- Training on D2, testing on D1, D2 and D3')
    model2 = SVC_model(X2_train, y2_train)
    print(get_results(model2, X2_train, y2_train))
    print('----- Training on D3, testing on D1, D2 and D3')
    model3 = SVC_model(X3_train, y3_train)
    print(get_results(model3, X3_train, y3_train))
    print("........................... RF ............................", flush=True)
    print('----- Training on D1, testing on D1, D2 and D3')
    model1 = RF_model(X1_train, y1_train)
    print(get_results(model1, X1_train, y1_train))
    print('----- Training on D2, testing on D1, D2 and D3')
    model2 = RF_model(X2_train, y2_train)
    print(get_results(model2, X2_train, y2_train))
    print('----- Training on D3, testing on D1, D2 and D3')
    model3 = RF_model(X3_train, y3_train)
    print(get_results(model3, X3_train, y3_train))
