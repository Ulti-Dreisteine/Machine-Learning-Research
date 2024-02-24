# -*- coding: utf-8 -*-
"""
Created on 2024/02/05 13:12:22

@File -> s0_read_data.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 读取数据
"""

import matplotlib.pyplot as plt
from typing import Optional
import seaborn as sns
import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    # ---- 数据加载 ---------------------------------------------------------------------------------
    
    label = "1"
    cols=["unit_nb", "time_cycle"] + ["set_1", "set_2", "set_3"] + [f"s_{i}" for i in range(1,22)]
    
    train_data = pd.read_csv(f"data/train_FD00{label}.txt", sep="\\s+", names=cols, header=None, 
                             index_col=False)
    
    test_data = pd.read_csv(f"data/test_FD00{label}.txt", sep="\\s+", names=cols, header=None, 
                            index_col=False)
    
    # train_data.head(500).to_csv("tmp.csv", index=False)
    
    # ---- 查看训练集中的寿命 -------------------------------------------------------------------------
    
    unit_nbs = np.unique(train_data["unit_nb"])
    
    for nb in unit_nbs:
        print(f"unit_nb: {nb}, life in cycles: {len(train_data[train_data['unit_nb'] == nb])}")
        
    # ---- 提取寿命最终时刻传感器数据 ------------------------------------------------------------------
    
    # 提取每台装备寿命初始时刻传感器数据
    init_life_data: Optional[pd.DataFrame] = None
    
    for nb in unit_nbs:
        d = train_data[train_data["unit_nb"] == nb].iloc[0, :].to_frame().T
        init_life_data = pd.concat([init_life_data, d], axis=0) if init_life_data is not None else d
    
    assert init_life_data is not None
    
    # 提取每台装备寿命最终时刻传感器数据
    final_life_data: Optional[pd.DataFrame] = None
    
    for nb in unit_nbs:
        d = train_data[train_data["unit_nb"] == nb].iloc[-1, :].to_frame().T
        final_life_data = pd.concat([final_life_data, d], axis=0) if final_life_data is not None else d
    
    assert final_life_data is not None
    
    # 查看对应指标寿命终点分布
    label = "s_4"
    sns.displot(init_life_data[label])
    sns.displot(final_life_data[label])
    plt.show()
    