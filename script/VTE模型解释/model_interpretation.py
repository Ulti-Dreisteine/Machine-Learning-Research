# -*- coding: utf-8 -*-
"""
Created on 2023/05/07 14:14:26

@File -> model_interpretation.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 模型解释
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from shap import TreeExplainer, Explanation
import shap
import pandas as pd
import numpy as np
import graphviz
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    
    ################################################################################################
    # 数据载入 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################

    data = pd.read_csv(f"{BASE_DIR}/data/data.csv")
    # data = pd.read_csv("data/data.csv")
    data.drop(["INPATIENT_ID", "SEX", "INJURY_TYPE", "INJURY_CAUSE"], axis=1, inplace=True)
    data = data.dropna(axis=0)

    # NOTE: 这里是Pandas表格形式而非Numpy数组
    X_df, Y_df = data[data.columns.drop("VTE")], data["VTE"]

    ################################################################################################
    # 数据集划分与建模 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, Y_df, test_size=0.3, shuffle=True, random_state=0)
    
    model = RandomForestClassifier(n_estimators=100,)
    model.fit(X_train, y_train)
    
    ################################################################################################
    # 模型解释 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    explainer = shap.Explainer(model)
    # shap_values = explainer.shap_values(X_train)
    shap_values = explainer(X_test)
    
    # shap.plots.waterfall(shap_values[0])
    shap.waterfall_plot(shap_values[1][:, 0], max_display=30)
    