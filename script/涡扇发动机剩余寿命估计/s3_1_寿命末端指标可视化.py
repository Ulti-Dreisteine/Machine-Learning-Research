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


if __name__ == "__main__":
    
    # ---- 载入实际数据 ------------------------------------------------------------------------------
    
    data = load_batch_test_data()
    unit_nbs = data["unit_nb"].unique()
    
    # ---- 绘制终寿值 --------------------------------------------------------------------------------
    
    final_life_values = []
    
    for unit_nb in unit_nbs:
        obs_samples = load_unit_test_data(data, unit_nb, "s_4")
        final_life_values.append(obs_samples[-1])
        
    # 计算统计指标
    ci_avg = np.mean(final_life_values)
    ci_median = np.percentile(final_life_values, 50)
    ci_lb = np.percentile(final_life_values, 5)
    ci_ub = np.percentile(final_life_values, 95)
    
    plt.figure(figsize=(8, 8))
    
    plt.subplot(2, 1, 2)
    plt.hist(final_life_values, bins=25, color="skyblue", edgecolor="black", density=True)
    plt.xlabel("final-life CI (S4) value")
    plt.ylabel("density")
    plt.grid(True, linewidth=0.5)

    plt.axvline(ci_avg, color="black", linestyle="-", label="mean", linewidth=3.0)
    plt.axvline(ci_median, color="green", linestyle="-", label="median", linewidth=3.0)
    plt.axvline(ci_lb, color="green", linestyle="--", label="CI (5% - 95%)")
    plt.axvline(ci_ub, color="green", linestyle="--")
    plt.title("distribution of final-life CI (S4) values of all engines", fontsize=16, fontweight="bold")
    plt.legend()
    
    # 生成matplotlib color map.
    cmap = plt.get_cmap("rainbow")
    
    plt.subplot(2, 1, 1)
    
    for unit_nb in unit_nbs:
        obs_samples = load_unit_test_data(data, unit_nb, "s_4")
        plt.plot(obs_samples, lw=0.8, alpha=0.8, c=cmap(unit_nb / len(unit_nbs)))
    
    # 获取xtick的最小最大值
    x_min, x_max = plt.xlim()
    plt.hlines(ci_avg, x_min, x_max, color="black", linestyle="-", linewidth=3.0, label="CI (avg. final value)")
    plt.legend()
    plt.grid(True, linewidth=0.5)
    plt.title("CI (S4) values of each engine", fontsize=16, fontweight="bold")
    plt.xlabel("number of cycles")
    plt.ylabel("CI (S4) value")
    
    plt.tight_layout()
    
    plt.savefig("img/寿命终值分布.png", dpi=450)
        
        