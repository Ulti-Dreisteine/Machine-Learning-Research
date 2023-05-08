from sklearn.neighbors import KDTree
from scipy.special import gamma
import numpy as np


def rand_x_values(x_ranges: list):
    """随机参数值. 参数分为自由参数和衍生参数两类"""
    x = [np.random.uniform(*x_ranges[i]) for i in range(len(x_ranges))]
    return np.array(x)


def cal_alpha(pdf_post_old, pdf_post):
    """计算接受概率. 注意这里分母为0时不要使用EPS, 因为分母量级可能为1E-100"""
    denominator = pdf_post_old
    return 1 if denominator == 0 else np.min([1, (pdf_post) / (pdf_post_old)])


# ---- K近邻操作 ------------------------------------------------------------------------------------

# 近邻样本查找
def _query_neighbors(tree, x, k):
    return tree.query(x, k=k)


def build_tree(samples, metric="euclidean"):
    tree = KDTree(samples, metric=metric)
    return tree


def search_nns(tree, sample_obs, k=None, max_nn_dist=None):
    k = tree.data.shape[0] if k is None else k
    nn_dists, nn_idxs = _query_neighbors(tree, sample_obs.reshape(1, -1), k=k)
    nn_dists, nn_idxs = nn_dists.flatten(), nn_idxs.flatten()
    
    # 按照距离阈值筛选样本
    if max_nn_dist is not None:
        _idxs = np.argwhere(nn_dists <= max_nn_dist).flatten()
        nn_idxs, nn_dists = nn_idxs[_idxs], nn_dists[_idxs]
    return nn_idxs, nn_dists
    

def build_tree_and_search_nns(samples, sample_obs, k=None, max_nn_dist=None, metric="euclidean"):
    tree = build_tree(samples, metric)
    nn_idxs, nn_dists = search_nns(tree, sample_obs, k, max_nn_dist)
    return nn_idxs, nn_dists, tree


def get_unit_ball_volume(d: int, metric: str = "euclidean"):
    """d维空间中按照euclidean或chebyshev距离计算所得的单位球体积"""
    if metric == "euclidean":
        return (np.pi ** (d / 2)) / gamma(1 + d / 2)  
    elif metric == "chebyshev":
        return 1
    else:
        raise ValueError(f"unsupported metric {metric}")
    

def cal_knn_prob_dens(tree_pop, sample, k, Np, dim, metric):
    _, nn_dists = search_nns(tree_pop, sample, k=k)
    cd = get_unit_ball_volume(dim, metric)
    pdf = k / Np / cd / (nn_dists[k - 1] ** dim)
    return pdf