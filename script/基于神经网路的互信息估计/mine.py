# -*- coding: utf-8 -*-
"""
Created on 2024/04/08 15:17:36

@File -> mine.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Mutual Information Neural Estimation
"""

import seaborn as sns
import numpy as np
import torch
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    # x数据是独立的，而y数据是相关的
    x = np.random.multivariate_normal(mean=[0,0], cov=[[1, 0],[0, 1]], size = 300)
    y = np.random.multivariate_normal(mean=[0,0], cov=[[1, 0.8],[0.8, 1]], size = 300)
    
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=x[:,0], y=x[:,1], label='x')
    sns.scatterplot(x=y[:,0], y=y[:,1], label='y')