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
    """
    对参数进行贝叶斯估计
    """
    
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
        trace = pm.sample(5000, chains=3, tune=1000, step=step)
        
        az.summary(trace, var_names=params2est, round_to=3)
    
    # 画图
    # plt.figure(figsize=(6, 8))
    
    # for i, param in enumerate(params2est):
    #     ax = plt.subplot(4, 1, i + 1)
    #     az.plot_forest(trace, var_names=[param], combined=False, figsize=(6, 4), show=False, ax=ax)
    #     plt.xticks(np.round(plt.xticks()[0], 4))
    
    # plt.tight_layout()
    
    # MAP估计
    map_estimate = pm.find_MAP(model=model)
    a_map = map_estimate["a"]
    b_map = map_estimate["b"]
    c_map = map_estimate["c"]
    sigma_obs_map = map_estimate["sigma_obs"]
    
    return float(a_map), float(b_map), float(c_map), float(sigma_obs_map)


# 重要性采样

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


def resample(pts, weights):
    """
    重要性采样：根据权重weights对粒子pts进行重采样
    """
    
    accum_weights = _get_accum_weights(weights)

    rand_nums = np.random.random(len(weights))
    pts_resampled = None
    for num in rand_nums:
        probs_sort = np.sort(np.append(accum_weights, num))
        insert_idx = np.argwhere(probs_sort == num)[0][0]
        
        pts_resampled = pts[insert_idx, :] if pts_resampled is None \
            else np.vstack((pts_resampled, pts[insert_idx, :]))
    
    return pts_resampled


if __name__ == "__main__":
    
    # ---- 载入实际数据 ------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    
    # ---- 设定参数 ---------------------------------------------------------------------------------
    
    unit_nb = 1
    x_col = "s_4"
    
    # ---- 提取所有装备寿命最终时刻传感器数据，用于后续寿命状态判断 -----------------------------------------
    
    finallife_x_samples = get_finallife_x_samples(data, x_col)
    
    plt.figure()
    plt.hist(finallife_x_samples, bins=10, density=True)
    
    # ---- 提取目标设备指标数据 -----------------------------------------------------------------------
    
    obs_samples = load_unit_test_data(data, unit_nb, x_col)[: 60]  # NOTE: 这里只观测前140个样本
    t_series = np.arange(len(obs_samples))
    N = len(t_series)
    
    # ---- 参数估计 ---------------------------------------------------------------------------------
    
    recal = True
    
    if recal:
        a_map, b_map, c_map, sigma_obs_map = estimate_params(t_series, obs_samples)
    else:
        a_map, b_map, c_map, sigma_obs_map = 0.001184, 1400.4486, -0.01703, 3.4064
    
    # ---- 基于粒子滤波的CI曲线估计 --------------------------------------------------------------------
    
    """
    这里通过粒子滤波而非参数估计获得所有的参数a、b、c、sigma_obs的估计值
    """
    
    n_marg_pts = 51  # 各边际上的颗粒数
    x_ranges = [[1395, 1450], [-0.1, 0.1]]  # 参数a、b、c、sigma_obs范围
    
    x_grids = np.meshgrid(
        *[np.linspace(x_range[0], x_range[1], n_marg_pts) for x_range in x_ranges]
    )
    
    # 根据离散化网格初始化粒子集
    init_pts = np.c_[*[x_grids[i].flatten() for i in range(len(x_ranges))]]
    
    # 逐时间步迭代
    pts_step_lst = []
    x_ft = None
    
    # TODO: 状态变量: a, x, dx, sigma
    # 第0步：a, b, c, sigma
    # 第 t >= 1 步：a, x_t, dx_t, sigma
    # 给出 t - 1 -> t 的迭代关系，进而计算权重，进而进行重采样
    
    for step in range(N):  # loc是当前位置时间步
        print(f"step = {step}, \ttotal = {N} \r", end="")
        
        pts_rs: np.ndarray
        
        # 获取当前步的粒子集
        if step == 0:
            pts = init_pts
        else:
            pts = pts_step_lst[-1]  # NOTE: 上一步重采样后的粒子集
        
        # 当前步状态值
        x_pts, dx_pts = pts[:, 0], pts[:, 1]

        # 计算该步的预测状态
        if step == 0:
            x_pred, dx_pred = x_pts, dx_pts
        else:
            x_pred, dx_pred = x_pts + dx_pts, dx_pts + a_map # 上一步重采样后粒子的迭代
        
        # 当前步观测值
        x_obs = obs_samples[step]
        
        # 计算所有粒子的权重
        w_pts = stats.norm.pdf(x_obs, x_pred, sigma_obs_map)
        
        # 更新粒子
        pts_udt = np.c_[x_pred, dx_pred]
        
        # 重采样
        pts_rs = resample(pts_udt, w_pts)
        
        # 记录此时间步上的粒子集
        pts_step_lst.append(pts_rs)
        
        # 记录本环节输入状态值期望 E[x_sys]
        Ex = np.mean(pts_rs, axis=0).reshape(1, 2)
        x_ft = Ex if x_ft is None else np.vstack((x_ft, Ex))
    
    # 画图
    plt.figure()
    plt.plot(obs_samples, label="obs")
    plt.plot(x_ft[:, 0], label="filtered")
        
    # ---- 进行外推 ---------------------------------------------------------------------------------
    
    N_pred = 140
    x_ext, x_std_ext = [], []
    init_pts = pts_step_lst[-1]
    pts_ext_lst = []
    
    for i in range(N_pred):
        if i == 0:
            pts = init_pts
        else:
            pts = pts_ext_lst[-1]
        
        # 上一步状态值
        x_pts, dx_pts = pts[:, 0], pts[:, 1]
        
        # 上一步重采样后粒子的迭代
        x_pred, dx_pred = x_pts + dx_pts, dx_pts + a_map
        
        # 更新粒子
        pts_udt = np.c_[x_pred, dx_pred]
        pts_ext_lst.append(pts_udt)
        
        x_ext.append(np.mean(pts_udt, axis=0))  # 本环节输入状态值期望 E[x_sys]
        x_std_ext.append(np.std(pts_udt, axis=0))
    
    x_ext, x_std_ext = np.array(x_ext), np.array(x_std_ext)
    
    # ---- 画图 -------------------------------------------------------------------------------------
    
    final_ci = 1431  # 最终寿命阈值
    
    # 总体分布
    plt.figure(figsize=(5, 6))
    
    plt.subplot(2, 1, 1)
    plt.scatter(
        range(len(obs_samples)), obs_samples, marker="o", s=12, c="w", edgecolors="k", lw=0.6, 
        zorder=2, label="obs. samples")
    plt.plot(range(N), x_ft[:, 0], "b-", zorder=1, label="filtered")
    plt.plot(range(N, N + N_pred), x_ext[:, 0], "r--", zorder=1, label="extrapolated")
    plt.xlabel("number of cycles $t$")
    plt.ylabel("CI, $x_{1,t}$")
    
    # 各粒子的轨迹
    for i in range(n_marg_pts ** 2):
        plt.plot([p[i, 0] for p in pts_step_lst], c="grey", alpha=0.003, lw=0.3, zorder=-2)
        
    plt.fill_between(
        range(N, N + N_pred), x_ext[:, 0] - 2 * x_std_ext[:, 0], x_ext[:, 0] + 2 * x_std_ext[:, 0], 
        color="red", alpha=0.3, zorder=-1)
    
    plt.hlines(final_ci, 0, N + N_pred, color="black", linestyle="--", label="CI threshold")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend(loc="upper left", fontsize=10)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(N), x_ft[:, 1], "b-", zorder=1, label="filtered")
    plt.plot(range(N, N + N_pred), x_ext[:, 1], "r--", zorder=1, label="extrapolated")
    plt.legend(loc="upper left", fontsize=10)
    plt.xlabel("number of cycles $t$")
    plt.ylabel("$\Delta$CI, $x_{2,t}$")
    
    # 各粒子的轨迹
    for i in range(n_marg_pts ** 2):
        plt.plot([p[i, 1] for p in pts_step_lst], c="grey", alpha=0.003, lw=0.3, zorder=-2)
    
    plt.fill_between(
        range(N, N + N_pred), x_ext[:, 1] - 2 * x_std_ext[:, 1], x_ext[:, 1] + 2 * x_std_ext[:, 1], 
        color="red", alpha=0.3, zorder=-1)
    
    plt.grid(True, linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("img/设备退化曲线.png", dpi=450)