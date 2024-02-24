import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), "../" * 2))
sys.path.insert(0, BASE_DIR)

from setting import plt


def load_batch_test_data(label: str = "1") -> pd.DataFrame:
    """
    载入批次试验数据
    
    Params:
    -------
    label: 批次标签
    """
    
    cols=["unit_nb", "time_cycle"] + ["set_1", "set_2", "set_3"] + [f"s_{i}" for i in range(1, 22)]
    data = pd.read_csv(
        f"data/train_FD00{label}.txt", sep="\\s+", names=cols, header=None, index_col=False)
    return data


def load_unit_test_data(data: pd.DataFrame, unit_nb: int, x_col: str) -> np.ndarray:
    """
    载入单台设备上的试验数据
    
    Params:
    -------
    data: 批次试验数据
    unit_nb: 设备编号
    x_col: 所采集的信号名
    """
    
    arr = data[data["unit_nb"] == unit_nb][x_col].values  # type: ignore
    return arr


if __name__ == "__main__":
    data = load_batch_test_data()
    ci_samples = load_unit_test_data(data, 1, "s_4")
    
    # 观测样本
    x_t0 = ci_samples[: -2].copy()                              # x_t
    dx_t0 = ci_samples[1: -1].copy() - ci_samples[: -2].copy()  # \Delta x_t
    x_t1 = ci_samples[1: -1].copy()                             # x_{t + 1}
    dx_t1 = ci_samples[2:].copy() - ci_samples[1: -1].copy()    # \Delta x_{t + 1}
    t0 = np.arange(len(x_t0))                                   # t
    t1 = t0 + 1                                                 # t + 1
    
    N = len(x_t0)
    
    # TODO: 降采样以降低样本间的关联
    
    # ---- 参数估计 ---------------------------------------------------------------------------------
    
    params2eval = [
        "a", "b", 
        "mu_s_1", "sigma_s_1", "mu_s_2", "sigma_s_2", 
        "sigma_o_1", "sigma_o_2"]
    
    with pm.Model() as model:
        # 给出各参数的先验分布
        a = pm.Normal("a", mu=1.0, sigma=0.1)
        b = pm.Normal("b", mu=1400.0, sigma=100.0)
        
        mu_s_1 = pm.Normal("mu_s_1", mu=0.0, sigma=0.5)
        sigma_s_1 = pm.HalfNormal("sigma_s_1", tau=5.0)
        mu_s_2 = pm.Normal("mu_s_2", mu=0.0, sigma=0.1)
        sigma_s_2 = pm.HalfNormal("sigma_s_2", tau=1.0)
        
        sigma_o_1 = pm.HalfNormal("sigma_o_1", tau=1.0)
        sigma_o_2 = pm.HalfNormal("sigma_o_2", tau=1.0)
        
        # 生成系统噪声样本
        eps_s_1 = pm.Normal("eps_s_1", mu=mu_s_1, sigma=sigma_s_1, shape=N)
        eps_s_2 = pm.Normal("eps_s_2", mu=mu_s_2, sigma=sigma_s_2, shape=N)
        
        # 生成观测噪声样本
        eps_o_1 = pm.Normal("eps_o_1", mu=0, sigma=sigma_o_1, shape=N)
        eps_o_2 = pm.Normal("eps_o_2", mu=0, sigma=sigma_o_2, shape=N)
        
        # 根据各变量观测值，建立似然关系
        x_t1_s = pm.Deterministic("x_t1_s", (x_t0 - eps_o_1) + (dx_t0 - eps_o_2) + eps_s_1)
        _ = pm.Normal(
            "x_t1_o",
            mu=x_t1_s, 
            sigma=sigma_o_1, 
            observed=x_t1)
        
        dx_t1_s = pm.Deterministic("dx_t1_s", (dx_t0 - eps_o_2) + a * t0 + eps_s_2)
        _ = pm.Normal(
            "dx_t1_o",
            mu=dx_t1_s, 
            sigma=sigma_o_2, 
            observed=dx_t1)
        
        # MCMC采样
        step = pm.Metropolis()
        # step = pm.NUTS()
        # step = pm.DEMetropolis()
        trace = pm.sample(1000, chains=3, tune=2000, step=step)
        
        az.summary(trace, var_names=params2eval)
        az.plot_posterior(trace, var_names=params2eval)
    
    # 总结后验
    az.plot_trace(
        trace,
        var_names=params2eval,
        compact=True,
        combined=True)
    plt.tight_layout()
    plt.show()
    