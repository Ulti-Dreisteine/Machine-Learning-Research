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
# import arviz as az
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_sys_output(x):
    return x + 1
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
    
    # ---- 产生模拟样本 ------------------------------------------------------------------------------

    # 噪声参数
    sys_noise_std = 0.1
    obs_noise_std = 10

    N = 100
    x_series = np.array([0.1])          # 状态值
    x_obs_series = np.array([0.1])      # 观测值

    for _ in range(N):
        xi = x_series[-1]
        sys_noise = np.random.normal(0, sys_noise_std)
        obs_noise = np.random.normal(0, obs_noise_std)
        yi = cal_sys_output(xi) + sys_noise
        y_obs = yi + obs_noise

        x_series = np.append(x_series, yi)
        x_obs_series = np.append(x_obs_series, y_obs)

    # plt.plot(np.arange(len(x_series)), x_series, c="k", lw=0.6)
    plt.figure(figsize=(5, 5))
    plt.scatter(np.arange(len(x_obs_series)), x_obs_series, marker="o", s=6, c="w", edgecolors="k", lw=0.6)
    # plt.show()

    # ---- 进行滤波 ---------------------------------------------------------------------------------

    n_particles = 1000
    # x_range = [np.min(x_obs_series), np.max(x_obs_series)]

    sys_noise_std = 0.1
    obs_noise_std = 10
    pf_rounds = 10 # PF迭代收敛次数 # TODO: 讨论是否需要迭代?

    x_filtered = []
    for loc in range(N):
        print(f"loc = {loc}, \ttotal = {N - 1} \r", end="")

        # 观测值
        y_obs = x_obs_series[loc + 1]
        x_range = [y_obs - 50, y_obs + 50]
        
        if loc == 0:
            particles = np.linspace(x_range[0], x_range[1], n_particles)

        # 前向并计算权重
        weights = np.array([])
        for x_sys in particles:
            sys_noise = np.random.normal(0, sys_noise_std)
            obs_noise = np.random.normal(0, obs_noise_std)

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

        x_filtered.append(np.mean(particles))
        
        # 更新粒子: 预测下一时刻的N个粒子
        particles = cal_sys_output(particles_resampled) + sys_noise
        
        # >>>>>>>>
        # for _ in range(pf_rounds):
        #     # 前向并计算权重
        #     weights = np.array([])
        #     for x_sys in particles:
        #         sys_noise = np.random.normal(0, sys_noise_std)
        #         obs_noise = np.random.normal(0, obs_noise_std)

        #         # 预测：前向计算
        #         y_sys = cal_sys_output(x_sys) + sys_noise

        #         # 校正：根据输出与观测的匹配程度，计算各粒子的重要度权重 wi = p(y_obs|xi)
        #         wi = stats.norm.pdf(y_obs, y_sys, obs_noise_std)
        #         weights = np.append(weights, wi)

        #     # 重要性采样
        #     accum_weights = _get_accum_weights(weights)

        #     rand_nums = np.random.random(n_particles)
        #     particles_resampled = np.array([])
        #     for num in rand_nums:
        #         probs_sort = np.sort(np.append(accum_weights, num))
        #         insert_idx = np.argwhere(probs_sort == num)[0][0]
        #         particles_resampled = np.append(particles_resampled, particles[insert_idx])

        #     # 更新粒子
        #     particles = particles_resampled

    xx = np.arange(len(x_filtered))
    yy = cal_sys_output(xx)
    
    plt.plot(xx, yy, "--", c="k", linewidth=2)
    plt.plot(x_filtered, linewidth=2)
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.grid(True, linewidth=0.5)
    plt.savefig("img/离散迭代滤波结果.png", dpi=450)
    plt.show()
        
    
    
    
    
    