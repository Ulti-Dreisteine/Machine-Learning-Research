# -*- coding: utf-8 -*-
"""
Created on 2023/05/06 16:03:30

@File -> resampling_regression.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于重采样的回归预测
"""

__doc__ = """
基于波士顿数据的重采样回归预测问题
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from scipy.special import gamma
from sklearn import datasets
import numpy as np
import arviz as az

from util import build_tree, rand_x_values, search_nns, cal_alpha, cal_knn_prob_dens

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_overall_pdfs(Y_norm, method="knn", k=None):
    if method == "kde":
        # 基于方差-偏差平衡和交叉验证进行带宽选择：
        # 带宽过窄将导致估计呈现高方差（即过拟合）；带宽过宽将导致估计呈现高偏差（即欠拟合）
        bw_cands = 10 ** np.linspace(-2, 2, 20)
        kde_est = KernelDensity(kernel="gaussian")
        grid = GridSearchCV(kde_est, {"bandwidth": bw_cands}, cv=LeaveOneOut())
        grid.fit(Y_norm)
        
        # 计算概率密度
        bw = grid.best_params_["bandwidth"]
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(Y_norm)
        Y_pdfs = np.exp(kde.score_samples(Y_norm))
    elif method == "knn":
        metric = "euclidean" # TODO: 需要交叉验证来确定k的最优值
        
        # 计算概率密度
        pop = Y_norm
        Np, dim = pop.shape
        tree_pop = build_tree(pop, metric)
        Y_pdfs = np.apply_along_axis(
            lambda x: cal_knn_prob_dens(tree_pop, x, k, Np, dim, metric), 1, Y_norm)
    else:
        ...
    
    return Y_pdfs


def show_pdfs(Y, pdfs):
    plt.figure()
    plt.fill_between(Y.flatten(), pdfs, alpha=0.6)
    plt.plot(
        Y.flatten(), np.full_like(Y.flatten(), -0.01), "|k", markeredgewidth=0.2, alpha=0.5
        )
    plt.ylim([0., 4.])
    plt.show()
    

# 接受-拒绝重采样
def resample(Y, Y_pdfs, m: float=10., max_rounds: int=10000, n_samples=None):
    N = Y.shape[0]
    idxs = np.arange(N)
    
    idxs_rs, Y_rs, weights_rs = np.array([]), np.array([]), np.array([])
    k = 0
    
    for i in range(max_rounds):
        idx_rand = np.random.choice(idxs)  # 从原分布中随机抽样
        
        y, pdf = Y[idx_rand], Y_pdfs[idx_rand]
        
        if i == 0:
            pass
        else:
            alpha = 1. / (pdf * m)  # 接受率, 等于预期采样分布密度除以原分布密度; 这里设定的分布为均匀分布
            if np.random.random() < alpha:  # 接受新样本, TODO 这一步有问题需要优化
                idxs_rs = np.append(idxs_rs, idx_rand)
                Y_rs = np.append(Y_rs, y)
                weights_rs = np.append(weights_rs, pdf)
                
                k += 1
                
                # 达到样本规模则中止循环
                if k == n_samples:
                    break
            else:
                continue
    
    return idxs_rs.astype(int), Y_rs, weights_rs


# 机器学习建模
def train_test_split(X, y, seed: int = None, test_ratio=0.2):
    X, y = X.copy(), y.copy()
    assert X.shape[0] == y.shape[0]
    assert 0 <= test_ratio < 1

    if seed is not None:
        np.random.seed(seed)
        shuffled_indexes = np.random.permutation(range(len(X)))
    else:
        shuffled_indexes = np.random.permutation(range(len(X)))

    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    return X[train_index], X[test_index], y[train_index], y[test_index]


if __name__ == "__main__":
    
    ################################################################################################
    # 数据载入 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    data = datasets.load_boston()
    X, Y = data["data"], data["target"].reshape(-1, 1)
    
    # 按照Y值排序
    idxs = np.argsort(Y.flatten())
    X, Y = X[idxs], Y[idxs]
    
    # 归一化
    scaler = MinMaxScaler()
    Y_norm = scaler.fit_transform(Y)
    
    # ---- 显示目标分布 -----------------------------------------------------------------------------
    
    # plt.figure(figsize=(6, 4))
    # az.plot_dist(Y_norm.flatten(), kind="hist")
    # plt.xlabel("house price in Boston")
    # plt.ylabel("prob. dens.")
    # plt.title(r"Distribution of $Y_{\rm norm}$", fontsize=16)
    # plt.show()
    
    ################################################################################################
    # 重采样建模计算 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
    
    method = "kde"  # 核密度估计"kde"或K近邻估计"knn"
    Y_pdfs = cal_overall_pdfs(Y_norm, method=method, k=100)
    
    # 画图展示
    # show_pdfs(Y_norm, Y_pdfs)
    
    # #### 接受-拒绝采样 ############################################################################
    
    idxs_rs, Y_rs, weights_rs = resample(Y_norm, Y_pdfs, n_samples=800)
    
    # 画图展示
    Y_rs = np.sort(Y_rs)
    pdfs_rs = cal_overall_pdfs(Y_rs.reshape(-1, 1), method="kde", k=200)
    # show_pdfs(Y_rs, pdfs_rs)
    
    # 合并画图
    plt.figure(figsize=(8, 5))
    plt.fill_between(Y_norm.flatten(), Y_pdfs.flatten(), alpha=0.6, label="original", color="b")
    plt.fill_between(Y_rs.flatten(), pdfs_rs.flatten(), alpha=0.6, label="resampled", color="r")
    plt.plot(
        Y_norm.flatten(), np.full_like(Y_norm.flatten(), -0.01), "|b", markeredgewidth=0.2, alpha=0.5
        )
    plt.plot(
        Y_rs.flatten(), np.full_like(Y_rs.flatten(), -0.02), "|r", markeredgewidth=0.2, alpha=0.5
        )
    plt.legend(loc="upper right")
    plt.ylim([0., 4.])
    plt.xlabel(r"$y$")
    plt.ylabel(r"prob. density $p(y)$")
    plt.grid()
    # plt.show()
    plt.savefig("重采样前后样本中目标值分布.png", dpi=450)
    
    ################################################################################################
    # 机器学习建模对比 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ################################################################################################
        
    model = RandomForestRegressor(n_estimators=100)
    rounds = 30
    
    errors = None
    for r in range(rounds):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y_norm.flatten(), test_ratio=0.2, seed=r)
        model.fit(X_train, Y_train.flatten())
        Y_test_pred = model.predict(X_test)
    
        # _e = np.c_[Y_test, np.abs(Y_test_pred - Y_test)]
        _e = np.c_[Y_test, Y_test_pred - Y_test]
        errors = _e if errors is None else np.r_[errors, _e]
    
    plt.figure(figsize=(8, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(errors[:, 0], errors[:, 1], c="b", marker="o",
                linewidth=0.5, alpha=0.3, s=12, label="original")
    plt.ylim([-0.7, 0.7])
    plt.xlabel(r"$y_{\rm pred}$")
    plt.ylabel(r"residual $y_{\rm pred} - y$")
    plt.grid()
    plt.legend(loc="upper right")
    
    X_rs, Y_rs = X[idxs_rs, :], Y_norm[idxs_rs, :]
    errors_rs = None
    for r in range(rounds):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_rs, Y_rs.flatten(), test_ratio=0.2, seed=r)
        model.fit(X_train, Y_train.flatten())
        Y_test_pred = model.predict(X_test)
    
        # _e = np.c_[Y_test, np.abs(Y_test_pred - Y_test)]
        _e = np.c_[Y_test, Y_test_pred - Y_test]
        errors_rs = _e if errors_rs is None else np.r_[errors_rs, _e]
    plt.subplot(1, 2, 2)
    plt.scatter(errors_rs[:, 0], errors_rs[:, 1], c="red", marker="o",
                linewidth=0.5, alpha=0.3, s=12, label="resampled")
    plt.ylim([-0.7, 0.7])
    plt.xlabel(r"$y_{\rm pred}$")
    plt.grid()
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig("重采样前后预测残差对比.png", dpi=450)

    
    