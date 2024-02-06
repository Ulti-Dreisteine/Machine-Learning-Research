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
    
    [x_{t+1} - b, dx_{t+1}]^T = [[1, 1], [a, 1]] \cdot [x_{t} - b, dx_{t}]^T + [eps_1, eps_2]^T
    
    则有：
    
    dx_{t+1} = a * x_{t} + dx_{t} - a * b + eps_2
    
    则建立贝叶斯参数估计模型 (x_{t}, dx_{t}) ~ dx_{t+1} 即可获得 a、b 系数值
    """

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import pytensor
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
    dx_tt = arr[2:].copy() - arr[1: -1].copy()
    
    X = np.c_[x_t, dx_t]
    y = dx_tt

    # ---- 贝叶斯参数估计 ----------------------------------------------------------------------------
    
    vars2eval = ["a", "b"]
    
    with pm.Model() as model:
        a = pm.Normal("a", mu=1.0, sigma=0.1)
        b = pm.Normal("b", mu=1300.0, sigma=100.0)
        
        
        mu = pm.Deterministic("mu", a * x_t + dx_t - a * b)
        
        _ = pm.Normal(
            "observed",
            mu=mu, 
            sigma=2, 
            observed=dx_tt)
        
        step = pm.NUTS()
        trace = pm.sample(10000, chains=1, tune=2000, step=step)
        
        az.summary(trace, var_names=vars2eval)
        az.plot_posterior(trace, var_names=vars2eval)
        
    # 总结后验
    az.plot_trace(
        trace,
        var_names=vars2eval,
        compact=True,
        combined=True)
    plt.tight_layout()
    plt.show()
    
    # ---- 结果验证 ---------------------------------------------------------------------------------
    
    # plt.figure()
    # ppc = pm.sample_posterior_predictive(trace, model, random_seed=42, progressbar=False)
    # az.plot_ppc(ppc)
    # plt.show()
    
    a = 0.043
    b = 1402
    
    y_true = dx_tt
    y_pred = a * x_t + dx_t - a * b
    
    # NOTE: 此结果显示线性回归模型存在系统误差，所以下文采用机器学习模型拟合
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=3, alpha=0.6)
    plt.scatter(y_true, y_pred - y_true, s=3, alpha=0.6)
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    plt.show()
    
    