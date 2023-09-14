# -*- coding: utf-8 -*-
"""
Created on 2023/09/14 11:31:22

@File -> 基于粒子滤波的输入分布估计.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于粒子滤波的输入分布估计
"""

from scipy import stats
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_sys_output(x):
    return 2 * x**3 + 3 * x**2 - 12 * x + 3
    # return x**2


def _get_accum_weights(weights):
    """将权重转化为累计值"""
    accum_weights = weights.copy()
    for i in range(len(weights) - 1):
        accum_weights[i] = np.sum(weights[: i + 1])
    accum_weights[-1] = np.sum(weights)
    accum_weights /= (accum_weights[-1])
    return accum_weights



if __name__ == "__main__":
    
    # ---- 使用PF对输入分布进行估计 -------------------------------------------------------------------
    
    y_obs = 10.0
    x = np.linspace(-5, 5, 100)
    y = cal_sys_output(x)
    
    # 噪声参数
    sys_noise_std = 1
    obs_noise_std = 2
    
    # 画图：显示观测值位置
    sys_noise_curve = np.random.normal(0, sys_noise_std, len(x))
    obs_noise_curve = np.random.normal(0, obs_noise_std, len(x))
    y_curve_obs = cal_sys_output(x) + sys_noise_curve + obs_noise_curve
    
    ax = plt.figure(figsize=(5, 5))
    plt.scatter(x, y_curve_obs, marker="o", s=6, c="w", edgecolors="k", lw=0.6)
    plt.plot(x, y, linewidth=1.0, c="k")
    plt.hlines(y_obs, -5, 5, colors="r", linewidth=1)
    plt.grid(True, linewidth=0.5)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.savefig("img/三次函数和观测值.png", dpi=450)
    plt.show()
    
    # 估计
    n_particles = 10000
    x_range = [-5, 5]
    pf_rounds = 1  # PF迭代收敛次数 # TODO: 讨论是否需要迭代?
    
    for round in range(pf_rounds):
        # 初始化粒子
        if round == 0:
            particles = np.linspace(x_range[0], x_range[1], n_particles)
        
        # 前向并计算权重
        weights = np.array([])
        for x_sys in particles:
            sys_noise = np.random.normal(0, sys_noise_std)
            
            # 预测：前向计算
            y_sys = cal_sys_output(x_sys) + sys_noise
            
            # 校正：根据输出与观测的匹配程度，计算各粒子的重要度权重 wi = p(y_obs|xi)
            wi = stats.norm.pdf(y_obs, y_sys, obs_noise_std)
            weights = np.append(weights, wi)
            
        # 重要性采样
        accum_weights = _get_accum_weights(weights)
        
        rand_nums = np.random.random(n_particles)
        particles_resampled = np.array([])
        for num in rand_nums:
            probs_sort = np.sort(np.append(accum_weights, num))
            insert_idx = np.argwhere(probs_sort == num)[0][0]
            particles_resampled = np.append(particles_resampled, particles[insert_idx])
        
        # 更新粒子
        particles = particles_resampled
    
    # 画图：显示估计效果
    plt.figure(figsize=(6, 5))
    plt.subplot(1, 2, 1)
    plt.hist(particles, range=x_range, bins=100, density=True)
    plt.xlabel("$x$")
    plt.ylabel(r"$p(x|y_{\rm obs})$")
    plt.grid(True, linewidth=0.5)
    # plt.show()
    
    # 计算输出估计
    y_sys = cal_sys_output(particles) + sys_noise
    
    # plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 2)
    plt.hist(y_sys, bins=50, density=True)
    plt.xlabel(r"$y_{\rm obs}$ predicted")
    plt.ylabel(r"$p(y_{\rm obs}|\bf{x}^{'})$")
    plt.grid(True, linewidth=0.5)
    
    
    plt.tight_layout()
    plt.savefig("img/三次函数输入后验分布估计结果.png", dpi=450)
    plt.show()
    
    
    