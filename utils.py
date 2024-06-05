
import jax.numpy as jnp
import numpy as np
import scipy as sp



def sample_n_sphere(n, k):

    x = np.random.normal(size=(k, n))
    x /= np.linalg.norm(x, 2, axis=1, keepdims=True)
    return x


def dists_on_sphere(x):

    k = x.shape[0]
    dist_mat = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                dist_mat[i, j] = -1
            else:
                dist_mat[i, j] = np.arccos(np.dot(x[i], x[j]))**2
    return dist_mat


def samp_dist_npoints(n, k, iters=100000):

    best_dist = 0
    for i in range(iters):
        points = sample_n_sphere(n, k)
        dists = dists_on_sphere(points)
        total_dist = jnp.min(dists[dists > 0])
        if total_dist > best_dist:
            best_dist = total_dist.copy()
            best_points = points
    return best_points


def l2_norm(W, axis=0):

    l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
    W = W / l2norm
    return W


def find_mat_thresh(dim, weight_range, iter4condthresh=10000,
                         cond_thresh_ratio=0.25, random_seed=0):

    random_seed = np.random.seed(random_seed)
    cond_list = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        W = np.random.uniform(weight_range[0], weight_range[1],
                              [dim, dim])
        W = l2_norm(W, 0)
        cond_list[i] = np.linalg.cond(W)
    cond_list.sort()
    cond_thresh = cond_list[int(iter4condthresh*cond_thresh_ratio)]
    return cond_thresh


def Sm_LeakyRelu(slope):

    return lambda x: smooth_leaky_relu(x, alpha=slope)


def smooth_leaky_relu(x, alpha=1.0):

    return alpha*x + (1 - alpha)*jnp.logaddexp(x, 0)


def match_cor(est_sources, true_sources, method="pearson"):

    dim = est_sources.shape[1]

    # calculate correlations
    if method == "pearson":
        corr = np.corrcoef(true_sources, est_sources, rowvar=False)
        corr = corr[0:dim, dim:]
    elif method == "spearman":
        corr, pvals = sp.stats.spearmanr(true_sources, est_sources)
        corr = corr[0:dim, dim:]

    # sort variables to try find matching components
    ridx, cidx = sp.optimize.linear_sum_assignment(-np.abs(corr))

    # calc with best matching components
    mean_abs_corr = np.mean(np.abs(corr[ridx, cidx]))
    s_est_sorted = est_sources[:, cidx]
    return mean_abs_corr, s_est_sorted, cidx


def match_indices(est_seq, true_seq):

    K = np.unique(est_seq).shape[0]
    match_counts = np.zeros((K, K), dtype=np.int)
    # algorithm to match estimated and true state indices
    for k in range(K):
        for l in range(K):
            est_k_idx = (est_seq == k).astype(np.int)
            true_l_idx = (true_seq == l).astype(np.int)
            match_counts[k, l] = -np.sum(est_k_idx == true_l_idx)
    _, matchidx = sp.optimize.linear_sum_assignment(match_counts)
    return matchidx


def cluster_accuracy(est_seq, true_seq):

    T = len(est_seq)
    # print(est_seq)
    matchidx = match_indices(est_seq, true_seq)
    for t in range(T):
        est_seq[t] = matchidx[est_seq[t]]
    return np.sum(est_seq == true_seq)/T
