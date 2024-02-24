# -*- coding: utf-8 -*-
"""
Created on 2024/02/24 14:16:40

@File -> s1_退化方程模拟.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 模拟退化方程
"""

from typing import Tuple, List
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import load_batch_test_data, load_unit_test_data


import numpy as np

def forward(a: float, b: float, c: float, sigma_obs: float, t_series: np.ndarray) \
    -> Tuple[List[float], List[float], List[float]]:
    """
    前向方程
    
    Params:
    -------
    a: float, 老化因子
    b: float, X的初始状态
    c: float, dX的初始值
    sigma_obs: float, 观测误差的标准差
    t_series: np.ndarray, 时间序列
    
    Returns:
    --------
    tuple: 包含观测值序列、X状态序列和dX状态序列的元组
    """
    
    x_states = [b]
    dx_states = [c]
    
    x_osvs = [b + np.random.normal(0, sigma_obs)]
    
    for t in t_series:
        x, dx = x_states[-1], dx_states[-1]
        
        x_next = x + dx
        dx_next = dx + a
        
        x_states.append(x_next)
        dx_states.append(dx_next)
        
        x_osvs.append(x_next + np.random.normal(0, sigma_obs))
    
    return x_osvs, x_states, dx_states
    

if __name__ == "__main__":
    
    # ---- 载入实际数据 ------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    obs_series = load_unit_test_data(data, 1, "s_4")
    
    # 时间记录
    t_series = np.arange(len(obs_series))
    
    # 画图
    plt.figure(figsize=(6, 5))
    plt.scatter(t_series, obs_series, label = "sample", s=12)
    plt.legend()
    plt.grid(linewidth=0.5, zorder=-1)
    plt.xlabel("number of cycles")
    plt.ylabel("CI value")
    plt.savefig("img/S4样本数据.png", dpi=450, bbox_inches="tight")
    
    # ---- 进行模拟 ---------------------------------------------------------------------------------
    
    a = 0.0015     # 与时间有关的老化因子
    b = 1400        # X的初始值
    c = 0           # dX的初始值
    sigma_obs = 5   # X的观测误差标准差
    
    x_osvs, x_states, dx_states = forward(a, b, c, sigma_obs, t_series)
        
    # ---- 画图对比模拟值与真实值 ----------------------------------------------------------------------
    
    """
    下图证明了退化方程结构的合理性, 将带入后续贝叶斯参数估计
    """
    
    plt.figure(figsize=(6, 5))
    plt.scatter(t_series, obs_series, label = "sample", s=12)
    plt.scatter(t_series, x_osvs[: -1], label = "simulated samples", marker="x", linewidths=1.0, s=12)
    plt.plot(x_states[: -1], label = "simulated curve", color = "orange", linewidth=2.0)
    plt.legend()
    plt.grid(linewidth=0.5, zorder=-1)
    plt.xlabel("number of cycles")
    plt.ylabel("CI value")
    plt.savefig("img/前向模拟数据.png", dpi=450, bbox_inches="tight")
        
    
    