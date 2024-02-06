# -*- coding: utf-8 -*-
"""
Created on 2024/02/05 15:36:53

@File -> s1_param_est.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 机理参数估计
"""

__doc__ = """
    如果：
    
    [x_{t+1} - b, dx_{t+1}]^T = [[1, 1], [a, 1]] \cdot [x_{t} - b, dx_{t}]^T 
    
    则有：
    
    dx_{t+1} = a * x_{t} + dx_{t} - a * b
    
    则建立贝叶斯参数估计模型 (x_{t}, dx_{t}) ~ dx_{t+1} 即可获得 a、b 系数值
    """

from sklearn.metrics import mean_squared_error as mse, explained_variance_score as evs,\
    r2_score as r2
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import pytensor
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 4))
sys.path.insert(0, BASE_DIR)

from setting import plt


def load_batch_test_data(label: str = "1") -> pd.DataFrame:
    """
    载入批次试验数据
    
    Params:
    -------
    label: 批次标签
    """
    
    cols=["unit_nb", "time_cycle"] + ["set_1", "set_2", "set_3"] + [f"s_{i}" for i in range(1, 22)]
    data = pd.read_csv(
        f"../data/train_FD00{label}.txt", sep="\\s+", names=cols, header=None, index_col=False)
    return data


def load_unit_test_data(data: pd.DataFrame, unit_nb: int, x_col: str) -> np.ndarray:
    """
    载入单台设备上的试验数据
    
    Params:
    -------
    data: 批次试验数据
    unit_nb: 设备编号
    x_col: 所采集的信号名
    """
    
    arr = data[data["unit_nb"] == unit_nb][x_col].values  # type: ignore
    return arr


# 机器学习建模
def _cal_metric(y_true, y_pred, metric: str):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    if metric == "r2":
        return r2(y_true, y_pred)
    if metric == "evs":
        return evs(y_true, y_pred)
    if metric == "mse":
        return mse(y_true, y_pred)
    if metric == "mape":
        idxs = np.where(y_true != 0)
        y_true = y_true[idxs]
        y_pred = y_pred[idxs]
        return np.sum(np.abs((y_pred - y_true) / y_true)) / len(y_true)


# 模型评价
def exec_model_test(X, y, model, metric: str="r2", test_ratio: float=0.3, rounds: int=10):
    """执行建模测试"""
    X, y = X.copy(), y.copy()
    N = X.shape[0]
    test_size = int(N * test_ratio)
    metrics = []
    for _ in range(rounds):
        shuffled_indexes = np.random.permutation(range(N))
        train_idxs = shuffled_indexes[test_size:]
        test_idxs = shuffled_indexes[:test_size]

        X_train, X_test = X[train_idxs, :], X[test_idxs, :]
        y_train, y_test = y[train_idxs], y[test_idxs]

        model.fit(X_train, y_train)
        m = _cal_metric(y_test, model.predict(X_test), metric)
        metrics.append(m)
    return np.mean(metrics), metrics
    

if __name__ == "__main__":
    
    # ---- 数据准备 ---------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    arr = load_unit_test_data(data, 1, "s_4")
    
    # 构建样本
    x_t = arr[: -2].copy()
    dx_t = arr[1: -1].copy() - arr[: -2].copy()
    dx_tt = arr[2:].copy() - arr[1: -1].copy()
    
    X = np.c_[x_t - 1400.0, dx_t]
    y = dx_tt

    # ---- 建立机器学习预测模型 -----------------------------------------------------------------------
    
    model = RandomForestRegressor(n_estimators=400, max_depth=5, random_state=0)
    
    N = X.shape[0]
    test_ratio = 0.3
    test_size = int(N * test_ratio)
    
    shuffled_indexes = np.random.permutation(range(N))
    train_idxs = shuffled_indexes[test_size:]
    test_idxs = shuffled_indexes[:test_size]

    X_train, X_test = X[train_idxs, :], X[test_idxs, :]
    y_train, y_test = y[train_idxs], y[test_idxs]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    m = _cal_metric(y_test, y_pred, "r2")
    
    # NOTE: 此结果显示线性回归模型存在系统误差，所以下文采用机器学习模型拟合
    plt.figure(figsize=(5, 5))
    plt.scatter(y_train, model.predict(X_train), s=3, alpha=0.6)
    plt.scatter(y_test, y_pred, s=3, alpha=0.6)
    # plt.scatter(y_test, y_pred - y_test, s=3, alpha=0.6)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.show()
    
    
    
    