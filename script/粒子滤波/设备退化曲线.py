from scipy import stats
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 3))
sys.path.insert(0, BASE_DIR)

from setting import plt

# 机理参数
A = np.array([[1, 1], [0, 1]])
B = np.array([[1, 0]])


def _get_accum_weights(weights):
    """将权重转化为累计值"""
    accum_weights = weights.copy()
    for i in range(len(weights) - 1):
        accum_weights[i] = np.sum(weights[: i + 1])
    accum_weights[-1] = np.sum(weights)
    accum_weights /= (accum_weights[-1])
    return accum_weights


def resample(particles, weights):
    """重要性采样"""
    accum_weights = _get_accum_weights(weights)

    rand_nums = np.random.random(len(weights))
    particles_resampled = None
    for num in rand_nums:
        probs_sort = np.sort(np.append(accum_weights, num))
        insert_idx = np.argwhere(probs_sort == num)[0][0]
        
        particles_resampled = particles[insert_idx, :] if particles_resampled is None \
            else np.vstack((particles_resampled, particles[insert_idx, :]))
    
    return particles_resampled


def gen_data(sys_noise_std, obs_noise_std, N):
    """生成模拟数据"""
    sys_noise = np.random.normal(0, sys_noise_std)
    obs_noise = np.random.normal(0, obs_noise_std)
    x_init = np.array([[8, -0.2]]).T  # 当前设备健康指数、健康指数退化速度
    y_init = np.dot(A, x_init) + sys_noise.reshape(2, 1)
    z_obs_init = np.dot(B, y_init) + obs_noise
    
    x_series = x_init            # 状态值
    z_obs_series = z_obs_init    # 观测值
    
    for _ in range(N - 1):
        sys_noise = np.random.normal(0, sys_noise_std)
        obs_noise = np.random.normal(0, obs_noise_std)
        
        xi = x_series[:, [-1]]
        yi = np.dot(A, xi) + sys_noise.reshape(2, 1)
        zi_obs = np.dot(B, yi) + obs_noise
        
        x_series = np.hstack((x_series, yi))
        z_obs_series = np.hstack((z_obs_series, zi_obs))
        
    return x_series, z_obs_series


if __name__ == "__main__":
    N = 40
    sys_noise_std = [0.1, 0.01]
    obs_noise_std = 0.1
    
    x_series, z_obs_series = gen_data( sys_noise_std, obs_noise_std, N)
    
    plt.figure(figsize=(5, 5))
    plt.plot(x_series[0, :])
    # plt.plot(x_series[1, :])
    plt.plot(z_obs_series[0, :])
    plt.show()
    
    # ---- 粒子滤波 ---------------------------------------------------------------------------------
    
    n_marg_particles = 41  # 各边际上的颗粒数
    sys_noise_std = [0.3, 0.03]
    obs_noise_std = 0.3
    
    x_range = [[0, 10], [-0.2, 0.1]]  # NOTE: 需要涵盖样本
    x_grids = np.meshgrid(
        np.linspace(x_range[0][0], x_range[0][1], n_marg_particles),
        np.linspace(x_range[1][0], x_range[1][1], n_marg_particles))
    
    x_filtered = None
    particles_lst = []
    for loc in range(N - 1):
        print(f"loc = {loc}, \ttotal = {N - 1} \r", end="")
        
        # 根据离散化网格初始化粒子集
        if loc == 0:
            particles = np.c_[x_grids[0].flatten(), x_grids[1].flatten()]
        
        particles_lst.append(particles)
            
        # 输出观测值
        z_obs = z_obs_series[:, loc + 1]
        
        # 前向并计算权重
        weights = np.array([])
        for xi in particles:
            xi = xi.reshape(2, 1)
            sys_noise = np.random.normal(0, sys_noise_std)
            obs_noise = np.random.normal(0, obs_noise_std)

            # 预测：前向计算
            yi = np.dot(A, xi) + sys_noise.reshape(2, 1)
            zi = np.dot(B, yi)

            # 校正：根据输出与观测的匹配程度，计算各粒子的重要度权重 wi = p(y_obs|xi)
            wi = stats.norm.pdf(z_obs.flatten(), zi.flatten(), obs_noise_std)[0]
            weights = np.append(weights, wi)
            
        # 重要性采样
        particles_resampled = resample(particles, weights)
        
        # 记录本环节输入状态值期望 E[x_sys]
        a = np.mean(particles_resampled, axis=0).reshape(2, 1)
        x_filtered = a if x_filtered is None else np.hstack((x_filtered, a))
        
        # 更新粒子: 预测下一时刻的N个粒子
        sys_noise = np.vstack((
            np.random.normal(0, sys_noise_std[0], n_marg_particles ** 2),
            np.random.normal(0, sys_noise_std[1], n_marg_particles ** 2)
        ))
        particles = np.dot(A, particles_resampled.T).T + sys_noise.T
    
    # ---- 进行外推 ---------------------------------------------------------------------------------
    
    N_pred = 30
    x_pred, x_std_pred = [], []
    for _ in range(N_pred):
        sys_noise = np.vstack((
            np.random.normal(0, sys_noise_std[0], n_marg_particles ** 2),
            np.random.normal(0, sys_noise_std[1], n_marg_particles ** 2)
        ))
        particles = np.dot(A, particles.T).T + sys_noise.T
        particles_lst.append(particles)
        
        x_pred.append(np.mean(particles, axis=0))  # 本环节输入状态值期望 E[x_sys]
        x_std_pred.append(np.std(particles, axis=0))
    
    x_pred, x_std_pred = np.array(x_pred), np.array(x_std_pred)
    
    # ---- 画图 -------------------------------------------------------------------------------------
    
    # 总体分布
    plt.figure(figsize=(5, 5))
    plt.subplot(2, 1, 1)
    plt.scatter(
        range(x_series.shape[1]), x_series[0, :], marker="o", s=12, c="w", edgecolors="k", lw=0.6, 
        zorder=2)
    plt.plot(range(N - 1), x_filtered[0, :], "b-", zorder=1)
    plt.plot(range(N - 1, N + N_pred - 1), x_pred[:, 0], "b--", zorder=1)
    plt.xlabel("time step $t$")
    plt.ylabel("$x_{1,t}$")
    
    # 各粒子的轨迹
    for i in range(n_marg_particles ** 2):
        plt.plot([p[i, 0] for p in particles_lst], c="grey", alpha=0.05, lw=0.3, zorder=-2)
    plt.ylim([-15, 10])
    plt.grid(True, linewidth=0.5, alpha=0.5)
    
    plt.subplot(2, 1, 2)
    plt.scatter(
        range(x_series.shape[1]), x_series[1, :], marker="o", s=12, c="w", edgecolors="k", lw=0.6,
        zorder=2)
    plt.plot(range(N - 1), x_filtered[1, :], "b-", zorder=1)
    plt.plot(range(N - 1, N + N_pred - 1), x_pred[:, 1], "b--", zorder=1)
    plt.xlabel("time step $t$")
    plt.ylabel("$x_{2,t}$")
    
    # 各粒子的轨迹
    for i in range(n_marg_particles ** 2):
        plt.plot([p[i, 1] for p in particles_lst], c="grey", alpha=0.05, lw=0.3, zorder=-2)
    plt.ylim([-0.8, 0.8])
    plt.grid(True, linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("img/设备退化曲线.png", dpi=450)
    plt.show()
    