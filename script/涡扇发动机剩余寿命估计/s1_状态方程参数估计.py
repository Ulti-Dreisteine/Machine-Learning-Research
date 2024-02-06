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
    x_t0 = ci_samples[: -2].copy()
    dx_t0 = ci_samples[1: -1].copy() - ci_samples[: -2].copy()
    x_t1 = ci_samples[1: -1].copy()
    dx_t1 = ci_samples[2:].copy() - ci_samples[1: -1].copy()
    
    # ---- 参数估计 ---------------------------------------------------------------------------------
    
    vars2eval = [
        "a", "b", "mu_sys_1", "sigma_sys_1", "mu_sys_2", "sigma_sys_2", "sigma_obs_1", "sigma_obs_2"]
    
    with pm.Model() as model:
        a = pm.Normal("a", mu=1.0, sigma=0.1)
        b = pm.Normal("b", mu=1300.0, sigma=100.0)
        
        # 系统噪声参数
        mu_sys_1 = pm.Normal("mu_sys_1", mu=0.0, sigma=0.5)
        sigma_sys_1 = pm.HalfNormal("sigma_sys_1", tau=5.0)
        mu_sys_2 = pm.Normal("mu_sys_2", mu=0.0, sigma=0.1)
        sigma_sys_2 = pm.HalfNormal("sigma_sys_2", tau=1.0)
        
        # 观测噪声参数
        sigma_obs_1 = pm.HalfNormal("sigma_obs_1", tau=1.0)
        sigma_obs_2 = pm.HalfNormal("sigma_obs_2", tau=1.0)
        
        # 观测噪声样本
        eps_obs_1 = pm.Normal("eps_obs_1", mu=0, sigma=sigma_obs_1, shape=len(x_t0))
        eps_obs_2 = pm.Normal("eps_obs_2", mu=0, sigma=sigma_obs_2, shape=len(x_t0))
        
        # x_t1的状态值
        x_t1_s = pm.Deterministic("x_t1_s", (x_t0 - eps_obs_1) + (dx_t0 - eps_obs_2))
        
        _ = pm.Normal(
            "x_t1_obs",
            mu=x_t1_s + mu_sys_1, 
            sigma=sigma_sys_1, 
            observed=x_t1)
        
        # dx_t1的状态值
        dx_t1_s = pm.Deterministic("dx_t1_s", a * (x_t0 - eps_obs_1 - b) + (dx_t0 - eps_obs_2))
        
        _ = pm.Normal(
            "dx_t1_obs",
            mu=dx_t1_s + mu_sys_2, 
            sigma=sigma_sys_2, 
            observed=dx_t1)
        
        # MCMC采样
        step = pm.Metropolis()
        # step = pm.NUTS()
        trace = pm.sample(10000, chains=1, tune=2000, step=step)
        
        az.summary(trace, var_names=vars2eval)
        az.plot_posterior(trace, var_names=vars2eval)
    
    # 总结后验
    az.plot_trace(
        trace,
        var_names=vars2eval,
        compact=True,
        combined=True)
    plt.tight_layout()
    plt.show()