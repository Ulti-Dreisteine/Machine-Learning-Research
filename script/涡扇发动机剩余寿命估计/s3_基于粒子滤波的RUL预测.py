# -*- coding: utf-8 -*-
"""
Created on 2024/02/26 14:16:27

@File -> s3_基于粒子滤波的RUL预测.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于粒子滤波的RUL预测
"""

from typing import Optional
from scipy import stats
import pytensor as pt
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt
from util import load_batch_test_data, load_unit_test_data


def get_finallife_x_samples(data: pd.DataFrame, x_col: str) -> np.ndarray:
    """
    提取所有发动机中x_col对应字段的最终寿命值
    """
    
    unit_nbs = np.unique(data["unit_nb"])
    final_life_data: Optional[pd.DataFrame] = None
    
    for nb in unit_nbs:
        d = data[data["unit_nb"] == nb].iloc[-1, :].to_frame().T
        final_life_data = pd.concat([final_life_data, d], axis=0) if final_life_data is not None else d
    
    assert final_life_data is not None
    final_x_samples = final_life_data[x_col].values
    
    return final_x_samples # type: ignore


def estimate_params(t_series: np.ndarray, obs_samples: np.ndarray) -> tuple[float, float, float, float]:
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
    
    # 画图
    plt.figure(figsize=(6, 8))
    
    for i, param in enumerate(params2est):
        ax = plt.subplot(4, 1, i + 1)
        az.plot_forest(trace, var_names=[param], combined=False, figsize=(6, 4), show=False, ax=ax)
        plt.xticks(np.round(plt.xticks()[0], 4))
    
    plt.tight_layout()
    
    # MAP估计
    map_estimate = pm.find_MAP(model=model)
    a_map = map_estimate["a"]
    b_map = map_estimate["b"]
    c_map = map_estimate["c"]
    sigma_obs_map = map_estimate["sigma_obs"]
    
    return float(a_map), float(b_map), float(c_map), float(sigma_obs_map)


def _get_accum_weights(weights):
    """
    将权重转化为累计值
    """
    
    accum_weights = weights.copy()
    for i in range(len(weights) - 1):
        accum_weights[i] = np.sum(weights[: i + 1])
    accum_weights[-1] = np.sum(weights)
    accum_weights /= (accum_weights[-1])
    
    return accum_weights


def resample(particles, weights):
    """
    重要性采样
    """
    
    accum_weights = _get_accum_weights(weights)

    rand_nums = np.random.random(len(weights))
    particles_resampled = None
    for num in rand_nums:
        probs_sort = np.sort(np.append(accum_weights, num))
        insert_idx = np.argwhere(probs_sort == num)[0][0]
        
        particles_resampled = particles[insert_idx, :] if particles_resampled is None \
            else np.vstack((particles_resampled, particles[insert_idx, :]))
    
    return particles_resampled


if __name__ == "__main__":
    
    # ---- 载入实际数据 ------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    obs_samples = load_unit_test_data(data, 1, "s_4")[:]
    
    t_series = np.arange(len(obs_samples))
    
    # ---- 设定参数 ---------------------------------------------------------------------------------
    
    unit_nb = 1
    x_col = "s_4"
    
    # ---- 提取所有装备寿命最终时刻传感器数据，用于后续寿命状态判断 -----------------------------------------
    
    finallife_x_samples = get_finallife_x_samples(data, x_col)
    
    plt.hist(finallife_x_samples, bins=10, density=True)
    
    # ---- 提取目标设备指标数据 -----------------------------------------------------------------------
    
    obs_samples = load_unit_test_data(data, unit_nb, x_col)[: 140]
    t_series = np.arange(len(obs_samples))
    
    # ---- 参数估计 ---------------------------------------------------------------------------------
    
    a_map, b_map, c_map, sigma_obs_map = estimate_params(t_series, obs_samples)
    
    # ---- 基于粒子滤波的CI曲线估计 --------------------------------------------------------------------
    
    A = ...
    B = ...
    
    obs_samples = obs_samples.reshape(1, -1)
    N = obs_samples.shape[1]
    
    n_marg_particles = 51           # 各边际上的颗粒数
    sys_noise_std = 0.0             # 系统误差
    obs_noise_std = sigma_obs_map   # 参数提醒: 参数估计
    
    x_range = [[-10, 100]]          # NOTE: 需要涵盖样本
    
    x_grids = np.meshgrid(
        np.linspace(x_range[0][0], x_range[0][1], n_marg_particles)
        )
    
    x_filtered = None
    particles_lst = []
    
    for loc in range(N - 1):
        print(f"loc = {loc}, \ttotal = {N - 1} \r", end="")
        
        # 根据离散化网格初始化粒子集
        if loc == 0:
            particles = np.c_[x_grids[0].flatten()]
        
        particles_lst.append(particles)
        
        # 输出观测值
        z_obs = obs_samples[:, loc + 1]
        
        # 前向并计算权重 
        weights = np.array([])
        
        b = np.array([[1400]])
        
        for xi in particles:
            xi = xi.reshape(2, 1)
            sys_noise = np.random.normal(0, sys_noise_std)
            obs_noise = np.random.normal(0, obs_noise_std)

            # 预测：前向计算
            yi = np.dot(A, xi) + sys_noise.reshape(2, 1)
            zi = np.dot(B, yi)

            # 校正：根据输出与观测的匹配程度，计算各粒子的重要度权重 wi = p(y_obs|xi)
            wi = stats.norm.pdf(z_obs.flatten(), zi.flatten(), obs_noise_std)[0]
            weights = np.append(weights, wi)
            
        # 重要性采样
        particles_resampled = resample(particles, weights)
        assert particles_resampled is not None
        
        # 记录本环节输入状态值期望 E[x_sys]
        a = np.mean(particles_resampled, axis=0).reshape(2, 1)
        x_filtered = a if x_filtered is None else np.hstack((x_filtered, a))
        
        # 更新粒子: 预测下一时刻的N个粒子
        sys_noise = np.vstack((
            np.random.normal(0, sys_noise_std[0], n_marg_particles ** 2),
            np.random.normal(0, sys_noise_std[1], n_marg_particles ** 2)
        ))
        particles = np.dot(A, particles_resampled.T).T + sys_noise.T