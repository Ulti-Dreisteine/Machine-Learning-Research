from scipy import stats
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt


def cal_sys_output(x):
    return x + 1


def _get_accum_weights(weights):
    """将权重转化为累计值"""
    accum_weights = weights.copy()
    for i in range(len(weights) - 1):
        accum_weights[i] = np.sum(weights[: i + 1])
    accum_weights[-1] = np.sum(weights)
    accum_weights /= (accum_weights[-1])
    return accum_weights


def gen_data(sys_noise_std, obs_noise_std, N):
    """生成模拟数据"""
    x_init = 1.1
    x_obs_init = x_init + np.random.normal(0, obs_noise_std)
    
    x_series = np.array([x_init])          # 状态值
    x_obs_series = np.array([x_obs_init])      # 观测值

    for _ in range(N - 1):
        xi = x_series[-1]
        sys_noise = np.random.normal(0, sys_noise_std)
        obs_noise = np.random.normal(0, obs_noise_std)
        yi = cal_sys_output(xi) + sys_noise
        y_obs = yi + obs_noise

        x_series = np.append(x_series, yi)
        x_obs_series = np.append(x_obs_series, y_obs)
        
    return x_series, x_obs_series


def resample(particles, weights):
    """重要性采样"""
    accum_weights = _get_accum_weights(weights)

    rand_nums = np.random.random(len(weights))
    particles_resampled = np.array([])
    for num in rand_nums:
        probs_sort = np.sort(np.append(accum_weights, num))
        insert_idx = np.argwhere(probs_sort == num)[0][0]
        particles_resampled = np.append(particles_resampled, particles[insert_idx])
    
    return particles_resampled
    

if __name__ == "__main__":
    
    # ---- 产生模拟样本 ------------------------------------------------------------------------------

    # 噪声参数
    sys_noise_std = 0.1
    obs_noise_std = 10
    
    N = 100
    
    x_series, x_obs_series = gen_data(sys_noise_std, obs_noise_std, N)
    
    # ---- 粒子滤波 ---------------------------------------------------------------------------------
    
    n_particles = 1000
    sys_noise_std = 0.1
    obs_noise_std = 10
    x_range = [-500, 500]  # NOTE: 需要同时涵盖样本和外推样本
    
    x_filtered = []
    for loc in range(N - 1):
        print(f"loc = {loc}, \ttotal = {N - 1} \r", end="")
        
        # 初始化粒子集
        if loc == 0:
            particles = np.linspace(x_range[0], x_range[1], n_particles)
        
        # 输出观测值
        y_obs = x_obs_series[loc + 1]
        
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
        particles_resampled = resample(particles, weights)
        
        # 记录结果
        x_filtered.append(np.mean(particles))  # 本环节输入状态值期望 E[x_sys]
        
        # 更新粒子: 预测下一时刻的N个粒子
        sys_noise = np.random.normal(0, sys_noise_std, n_particles)
        particles = cal_sys_output(particles_resampled) + sys_noise
    
    # 最后一个样本的滤波值
    x_filtered.append(np.mean(particles))
    
    # ---- 进行外推 ---------------------------------------------------------------------------------
    
    N_pred = 200
    x_pred, x_pred_std = [], []
    for _ in range(N_pred):
        sys_noise = np.random.normal(0, sys_noise_std, n_particles)
        particles = cal_sys_output(particles) + sys_noise
        
        x_pred.append(np.mean(particles))  # 本环节输入状态值期望 E[x_sys]
        x_pred_std.append(np.std(particles))
    
    # 真实关系    
    xx = np.arange(N + N_pred)
    yy = cal_sys_output(xx)
    
    x_pred, x_pred_std = np.array(x_pred), np.array(x_pred_std)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(np.arange(N), x_obs_series, marker="o", s=6, c="w", edgecolors="k", lw=0.6)
    plt.plot(xx, yy, "--", c="k", linewidth=2)
    plt.plot(x_filtered, linewidth=2)
    # plt.plot(np.arange(N, N + N_pred), x_pred, linewidth=2, color="r")
    plt.fill_between(np.arange(N, N + N_pred), x_pred - 3 * x_pred_std, x_pred + 3 * x_pred_std, alpha=0.3)
    plt.xlabel("$t$")
    plt.ylabel("$x$")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.savefig("img/离散迭代滤波结果.png", dpi=450)
    plt.show()
        
        
    


    