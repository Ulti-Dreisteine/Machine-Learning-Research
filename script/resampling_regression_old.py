# -*- coding: utf-8 -*-
"""
Created on 2023/05/03 15:44:32

@File -> resampling_regression.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 回归问题的重采样
"""

__doc__ = """
基于波士顿数据的重采样回归预测问题
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from scipy.special import gamma
from sklearn import datasets
import numpy as np
import arviz as az

from util import build_tree, rand_x_values, search_nns, cal_alpha

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


# 单位球体积
def _get_unit_ball_volume(d: int, metric: str = "euclidean"):
    """d维空间中按照euclidean或chebyshev距离计算所得的单位球体积"""
    if metric == "euclidean":
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == "chebyshev":
        return 1
    else:
        raise ValueError(f"unsupported metric {metric}")
    
    
def cal_knn_prob_dens(tree_pop, sample, k, Np, dim, metric):
    _, nn_dists = search_nns(tree_pop, sample, k=k)
    cd = _get_unit_ball_volume(dim, metric)
    pdf = k / Np / cd / (nn_dists[k - 1] ** dim)
    return pdf


def cal_overall_pdfs(Y_norm, method="knn"):
    if method == "kde":
        # 基于方差-偏差平衡和交叉验证进行带宽选择：带宽过窄将导致估计呈现高方差(即过拟合)；带宽过宽将导致估计呈现高偏差(即欠拟合)
        bw_cands = 10 ** np.linspace(-2, 1, 20)
        kde_est = KernelDensity(kernel="gaussian")
        grid = GridSearchCV(kde_est, {"bandwidth": bw_cands}, cv=LeaveOneOut())
        grid.fit(Y_norm)
        
        # 计算概率密度
        bw = grid.best_params_["bandwidth"]
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(Y_norm)
        Y_pdfs = np.exp(kde.score_samples(Y_norm))
    elif method == "knn":
        metric = "euclidean"
        k = 50  # TODO: 需要交叉验证来确定最优值
        
        # 计算概率密度
        pop = Y_norm
        Np, dim = pop.shape
        tree_pop = build_tree(pop, metric)
        Y_pdfs = np.apply_along_axis(
            lambda x: cal_knn_prob_dens(tree_pop, x, k, Np, dim, metric), 1, Y_norm)
    else:
        ...
    
    return Y_pdfs
    

if __name__ == "__main__":
    
    ################################################################################################
    # 数据载入 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    data = datasets.load_boston()
    X, Y = data["data"], data["target"].reshape(-1, 1)
    
    idxs = np.argsort(Y.flatten())
    X, Y = X[idxs], Y[idxs]
    
    ################################################################################################
    # 重采样建模计算 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    scaler = MinMaxScaler()
    Y_norm = scaler.fit_transform(Y)
    
    # ---- 显示目标分布 -----------------------------------------------------------------------------
    
    plt.figure(figsize=(6, 4))
    az.plot_dist(Y_norm.flatten(), kind="hist")
    plt.xlabel("house price in Boston")
    plt.ylabel("prob. dens.")
    plt.title(r"Distribution of $Y_{\rm norm}$", fontsize=16)
    plt.show()
    
    # #### 1. 总体样本概率密度估计 ###################################################################
    
    method = "knn"  # 核密度估计"kde"或K近邻估计"knn"
    Y_pdfs = cal_overall_pdfs(Y_norm, method=method)
    
    # 画图展示
    plt.figure()
    plt.fill_between(Y_norm.flatten(), Y_pdfs, alpha=0.6)
    plt.plot(
        Y_norm.flatten(), np.full_like(Y_norm.flatten(), -0.01), "|k", markeredgewidth=0.1
        )
    
    # #### 接受拒绝采样 ##############################################################################
    
    # 通过接受拒绝采样调整目标分布
    N = Y_norm.shape[0]
    idxs = np.arange(N)
    m = 5
    
    rounds = 2000
    Y_resampled = np.array([])
    weights_resampled = np.array([])
    for i in range(rounds):
        idx_rand = np.random.randint(N)
        
        y = Y_norm[idx_rand]
        pdf = Y_pdfs[idx_rand]
        
        if i == 0:
            pass
        else:
            # alpha = cal_alpha(weights_resampled[-1], pdf)
            # alpha = cal_alpha(pdf, 1)
            alpha = 1. / (pdf * m)  # 接受率
            if np.random.random() < alpha:  # 接受新样本
                Y_resampled = np.append(Y_resampled, y)
                weights_resampled = np.append(weights_resampled, pdf)
                pass
            else:
                # y, pdf = Y_resampled[-1], weights_resampled[-1]
                # y = Y_resampled[-1]
                continue
                
    Y_resampled = np.sort(Y_resampled)
    pdfs_resampled = cal_overall_pdfs(Y_resampled.reshape(-1, 1), method="kde")
    
    # 画图展示
    plt.figure()
    plt.fill_between(Y_resampled.flatten(), pdfs_resampled, alpha=0.6)
    plt.plot(
        Y_resampled.flatten(), np.full_like(Y_resampled.flatten(), -0.01), "|k", markeredgewidth=0.1
        )
        
    
    
    
    
    
    