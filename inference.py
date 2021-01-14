#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 8/1/2021 4:51 PM
# @Author  : Feng
# @FileName: inference.py
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
import GlobalPrameters as g
def predict():
    basicModels=[]
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=23)
    for i in range(5):
        basicModels.append(joblib.load(g.modelSavePath+'/basic_{}.pkl'.format(i)))
    metaModel=joblib.load(g.modelSavePath+"/meta.pkl")
    predictions = None
    for basicModel in basicModels:
        y_pred = basicModel.predict(x_test)
        if isinstance(predictions, np.ndarray):
            predictions = np.vstack((predictions, y_pred))
        else:
            predictions = y_pred
    predictions = predictions.T
    out = metaModel.score(predictions,y_test)
    print(out)
if __name__ == '__main__':
        predict()
