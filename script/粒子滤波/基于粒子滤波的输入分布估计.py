# -*- coding: utf-8 -*-
"""
Created on 2023/09/13 15:27:01

@File -> 基于粒子滤波的输入分布估计.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于粒子滤波的非线性系统输入分布估计
"""

import numpy as np
import random
from scipy import stats
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_sys_output(x):
    return 2 * x**3 + 3 * x**2 - 12 * x + 3


def _get_accum_probs(weights):
    """
    将适应度转化为累计概率
    :param weights: np.array, 一维适应度表序列
    :return: accum_prob: np.array, 一维累计概率表
    """
    accum_weights = weights.copy()
    for i in range(len(weights) - 1):
        accum_weights[i] = np.sum(weights[: i + 1])
    accum_weights[-1] = np.sum(weights)
    accum_probs = accum_weights / (accum_weights[-1])
    return accum_probs


if __name__ == "__main__":
    x = np.linspace(-5, 5, 100)
    noise = np.random.normal(0, 0.01, len(x))   # 参数提醒：过程噪声
    y = cal_sys_output(x) + noise

    plt.plot(x, y)

    # ---- 基于粒子滤波的分布估计 ---------------------------------------------------------------------

    rounds = 10
    x_range = [-5, 5]
    n_particles = 1000
    y_obs = 0.0

    x_particles_records = []
    for i in range(rounds):
        if i == 0:
            x_particles = np.linspace(x_range[0], x_range[1], n_particles)
        
        x_particles_records.append(x_particles)
        
        # 前向并计算权重
        weights = np.array([])
        for xi in x_particles:
            noise = np.random.normal(0, 0.1)       # 参数提醒：过程噪声

            # 预测：前向计算
            yi = cal_sys_output(xi) + noise

            # 校正：根据输出与观测的匹配程度，计算各粒子的重要度权重 wi = p(y_obs|xi)
            wi = stats.norm.pdf(yi, y_obs, 0.1)    # 参数提醒：观测噪声
            weights = np.append(weights, wi)

        # 重采样
        accum_probs = _get_accum_probs(weights)

        rand_nums = np.random.random(n_particles)
        x_particles_new = np.array([])
        for num in rand_nums:
            probs_sort = np.sort(np.append(accum_probs, num))
            insert_idx = np.argwhere(probs_sort == num)[0][0]
            x_particles_new = np.append(x_particles_new, x_particles[insert_idx])

        x_particles = x_particles_new
        
    # ---- 粒子滤波结果分析 --------------------------------------------------------------------------
    
    plt.figure(figsize=(8, 5))
    for i in range(rounds):
        x_particles = x_particles_records[i]
        plt.scatter(
            np.ones_like(x_particles) * i, x_particles, alpha=0.1, color="k", marker="x", s=6)
        
    plt.figure(figsize=(5, 5))
    plt.hist(x_particles_records[-1], bins=50)
        
    
        
    
    