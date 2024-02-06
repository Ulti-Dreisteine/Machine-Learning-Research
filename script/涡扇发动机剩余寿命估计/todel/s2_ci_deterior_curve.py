# -*- coding: utf-8 -*-
"""
Created on 2024/02/05 14:29:23

@File -> s1_ci_deterior_curve.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 绘制CI剩余寿命估计曲线
"""

import matplotlib.pyplot as plt
from typing import Optional
from scipy import stats
import pandas as pd
import numpy as np


def get_final_x_samples(data: pd.DataFrame, x_col: str) -> np.ndarray:
    unit_nbs = np.unique(data["unit_nb"])
    final_life_data: Optional[pd.DataFrame] = None
    
    for nb in unit_nbs:
        d = data[data["unit_nb"] == nb].iloc[-1, :].to_frame().T
        final_life_data = pd.concat([final_life_data, d], axis=0) if final_life_data is not None else d
    
    assert final_life_data is not None
    final_x_samples = final_life_data[x_col].values
    
    return final_x_samples # type: ignore


# 机理参数
A = np.array([[1, 1], [0.47, 1]])  # 参数提醒: 参数估计
B = np.array([[1, 0]])


def _get_accum_weights(weights):
    """将权重转化为累计值"""
    accum_weights = weights.copy()
    for i in range(len(weights) - 1):
        accum_weights[i] = np.sum(weights[: i + 1])
    accum_weights[-1] = np.sum(weights)
    accum_weights /= (accum_weights[-1])
    return accum_weights


def resample(particles, weights):
    """重要性采样"""
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
    
    # ---- 数据加载 ---------------------------------------------------------------------------------
    
    label = "1"
    cols=["unit_nb", "time_cycle"] + ["set_1", "set_2", "set_3"] + [f"s_{i}" for i in range(1,22)]
    
    data = pd.read_csv(f"data/train_FD00{label}.txt", sep="\\s+", names=cols, header=None, 
                       index_col=False)
    
    unit_nbs = np.unique(data["unit_nb"])
    
    # ---- 设定参数 ---------------------------------------------------------------------------------
    
    unit_nb = 1
    x_col = "s_4"
    
    # ---- 提取所有装备寿命最终时刻传感器数据，用于后续寿命状态判断 -----------------------------------------
    
    final_x_samples = get_final_x_samples(data, x_col)
    
    # ---- 提取目标设备指标数据 -----------------------------------------------------------------------
    
    arr = data[data["unit_nb"] == unit_nb][x_col].values.reshape(1, -1)  # type: np.ndarray
    arr = arr[:, :140]
    arr -= 1407.0
    
    # ---- 基于粒子滤波的CI曲线估计 --------------------------------------------------------------------
    
    N = arr.shape[1]
    
    n_marg_particles = 51  # 各边际上的颗粒数
    sys_noise_std = [0.36, 5.6]     # 参数提醒: 参数估计
    obs_noise_std = 3               # 参数提醒: 参数估计
    
    x_range = [[-10, 100], [-10.0, 10.0]]  # NOTE: 需要涵盖样本
    x_grids = np.meshgrid(
        np.linspace(x_range[0][0], x_range[0][1], n_marg_particles),
        np.linspace(x_range[1][0], x_range[1][1], n_marg_particles))
    
    x_filtered = None
    particles_lst = []
    
    for loc in range(N - 1):
        print(f"loc = {loc}, \ttotal = {N - 1} \r", end="")
        
        # 根据离散化网格初始化粒子集
        if loc == 0:
            particles = np.c_[x_grids[0].flatten(), x_grids[1].flatten()]
        
        particles_lst.append(particles)
            
        # 输出观测值
        z_obs = arr[:, loc + 1]
        
        # 前向并计算权重
        weights = np.array([])
        b = np.array([[1400], [0]])
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
    
    plt.plot(x_filtered[0, :].flatten())
    plt.plot(arr[0, :].flatten())
    plt.ylim([-5, 15])
    plt.show()
        
    # ---- 进行外推 ---------------------------------------------------------------------------------
    
    N_pred = 50
    x_pred, x_std_pred = [], []
    for _ in range(N_pred):
        sys_noise = np.vstack((
            np.random.normal(0, sys_noise_std[0], n_marg_particles ** 2),
            np.random.normal(0, sys_noise_std[1], n_marg_particles ** 2)
        ))
        particles = np.dot(A, particles.T).T + sys_noise.T
        particles_lst.append(particles)
        
        x_pred.append(np.mean(particles, axis=0))  # 本环节输入状态值期望 E[x_sys]
        x_std_pred.append(np.std(particles, axis=0))
    
    x_pred, x_std_pred = np.array(x_pred), np.array(x_std_pred)
    
    plt.plot(list(x_filtered[0, :].flatten()) + list(x_pred[:, 0]))
    plt.show()
    
    # ---- 画图 -------------------------------------------------------------------------------------
    
    # 总体分布
    plt.figure(figsize=(5, 5))
    plt.subplot(2, 1, 1)
    # plt.scatter(
    #     range(x_series.shape[1]), x_series[0, :], marker="o", s=12, c="w", edgecolors="k", lw=0.6, 
    #     zorder=2)
    plt.plot(range(N - 1), x_filtered[0, :], "b-", zorder=1)
    plt.plot(range(N - 1, N + N_pred - 1), x_pred[:, 0], "b--", zorder=1)
    plt.xlabel("time step $t$")
    plt.ylabel("$x_{1,t}$")
    
    # 各粒子的轨迹
    for i in range(n_marg_particles ** 2):
        plt.plot([p[i, 0] for p in particles_lst], c="grey", alpha=0.05, lw=0.3, zorder=-2)
    plt.ylim([-10, 50])
    plt.grid(True, linewidth=0.5, alpha=0.5)
    
    plt.subplot(2, 1, 2)
    # plt.scatter(
    #     range(x_series.shape[1]), x_series[1, :], marker="o", s=12, c="w", edgecolors="k", lw=0.6,
    #     zorder=2)
    plt.plot(range(N - 1), x_filtered[1, :], "b-", zorder=1)
    plt.plot(range(N - 1, N + N_pred - 1), x_pred[:, 1], "b--", zorder=1)
    plt.xlabel("time step $t$")
    plt.ylabel("$x_{2,t}$")
    
    # 各粒子的轨迹
    for i in range(n_marg_particles ** 2):
        plt.plot([p[i, 1] for p in particles_lst], c="grey", alpha=0.05, lw=0.3, zorder=-2)
    # plt.ylim([0, 15])
    plt.grid(True, linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    # plt.savefig("img/设备退化曲线.png", dpi=450)
    plt.show()
    
    
    