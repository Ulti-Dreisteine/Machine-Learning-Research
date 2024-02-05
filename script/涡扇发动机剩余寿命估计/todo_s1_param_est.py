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

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
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
        f"data/train_FD00{label}.txt", sep="\\s+", names=cols, header=None, index_col=False)
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
    

if __name__ == "__main__":
    
    # ---- 数据准备 ---------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    arr = load_unit_test_data(data, 1, "s_4")
    
    # 构建样本
    x_t = arr[: -2].copy()
    dx_t = arr[1: -1].copy() - arr[: -2].copy()
    dx_tt = arr[2:].copy() - arr[: -2].copy()
    
    X = np.c_[x_t, dx_t]
    y = dx_tt

    # ---- 贝叶斯参数估计 ----------------------------------------------------------------------------
    
    # 基于X和y的观测数据，通过参数估计获得 a、b 系数值
    
    # model = LinearRegression(fit_intercept=True)
    # model.fit(X, y)
    
    
    