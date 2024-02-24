# -*- coding: utf-8 -*-
"""
Created on 2024/02/24 14:42:18

@File -> s2_退化方程参数估计.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 退化方程参数估计
"""

import numpy as np
import pymc as pm
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import load_batch_test_data, load_unit_test_data

if __name__ == "__main__":
    
    # ---- 载入实际数据 ------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    obs_samples = load_unit_test_data(data, 1, "s_4")
    
    t_series = np.arange(len(obs_samples))
    
    # ---- 参数估计 ---------------------------------------------------------------------------------
    
    params2est = [
        "a",            # 与时间有关的老化因子
        "b",            # X的初始状态
        "c",            # dX的初始值
        "sigma_obs",    # X的观测误差标准差
        ]
    
    with pm.Model() as model:
        a = pm.Uniform("a", lower=0.0, upper=0.1)
        b = pm.Uniform("b", lower=1000, upper=2000)
        c = pm.Uniform("c", lower=-1., upper=1.)
        sigma_obs = pm.HalfNormal("sigma_obs", tau=1.0)
        
        