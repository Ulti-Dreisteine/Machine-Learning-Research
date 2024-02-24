# -*- coding: utf-8 -*-
"""
Created on 2024/02/24 14:42:18

@File -> s2_退化方程参数估计.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 退化方程参数估计
"""

import pytensor as pt
import arviz as az
import numpy as np
import pymc as pm
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import load_batch_test_data, load_unit_test_data

from s1_退化方程正向模拟 import forward

if __name__ == "__main__":
    
    # ---- 载入实际数据 ------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    obs_samples = load_unit_test_data(data, 1, "s_4")[:]
    
    t_series = np.arange(len(obs_samples))
    
    # ---- 参数估计 ---------------------------------------------------------------------------------
    
    params2est = [
        "a",            # 老化因子
        "b",            # X的初始状态
        "c",            # dX的初始值
        "sigma_obs",    # X的观测误差标准差
        ]
    
    with pm.Model() as model:
        a = pm.Uniform("a", lower=0.0, upper=0.1)
        b = pm.Uniform("b", lower=1000, upper=2000)
        c = pm.Uniform("c", lower=-1., upper=1.)
        sigma_obs = pm.HalfNormal("sigma_obs", tau=1.0)
        
        dX = pm.Deterministic("dX", a * t_series + c)
        X = pm.Deterministic("X", pt.tensor.dot(np.tri(len(t_series)), dX.reshape(-1, 1)) + b)
        
        X_obs = pm.Normal("X_obs", X, sigma_obs, observed=obs_samples)
        
        # MCMC采样
        step = pm.NUTS()
        trace = pm.sample(5000, chains=4, tune=1000, step=step)
        
        az.summary(trace, var_names=params2est, round_to=3)
    
    pm.model_graph.model_to_graphviz(model)
    
    # NOTE: az.plot_trace()对于多参数的计算速度很慢
    # with model:
    #     az.plot_trace(trace, var_names=["a"], combined=True)
    
    plt.figure(figsize=(6, 8))
    
    for i, param in enumerate(params2est):
        ax = plt.subplot(4, 1, i + 1)
        az.plot_forest(trace, var_names=[param], combined=False, figsize=(6, 4), show=False, ax=ax)
        plt.xticks(np.round(plt.xticks()[0], 4))
    
    plt.tight_layout()
    
    # ---- MAP估计 ----------------------------------------------------------------------------------
    
    map_estimate = pm.find_MAP(model=model)
    a_map = map_estimate["a"]
    b_map = map_estimate["b"]
    c_map = map_estimate["c"]
    sigma_obs_map = map_estimate["sigma_obs"]
    
    # ---- 基于参数估计结果的正向模拟 ------------------------------------------------------------------
    
    # 时间记录
    t_series = np.arange(len(obs_samples))
    
    x_osvs, x_states, dx_states = forward(a_map, b_map, c_map, sigma_obs_map, t_series)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(t_series, obs_samples, label = "sample", s=12)
    plt.scatter(t_series, x_osvs[: -1], label = "simulated samples", marker="x", linewidths=1.0, s=12)
    plt.plot(x_states[: -1], label = "simulated curve", color = "orange", linewidth=2.0)
    plt.legend()
    plt.grid(linewidth=0.5, zorder=-1)
    plt.xlabel("number of cycles")
    plt.ylabel("CI value")
    plt.savefig("img/参数估计结果模拟.png", dpi=450, bbox_inches="tight")
        
        
        
        
        
        
        