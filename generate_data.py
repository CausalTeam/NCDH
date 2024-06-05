
import numpy as np
import scipy as sp
from scipy import linalg
from utils import samp_dist_npoints


def gen_data(num_comp, num_latent_states, sequence_length,
                    state_stickiness=0.99, radius=2, random_seed=0):

    # set random see
    np.random.seed(random_seed)

    # create transition matrix
    transition_matrix = np.zeros((num_latent_states,
                                  num_latent_states))
    for i in range(num_latent_states):
        for j in range(num_latent_states):
            if j == i:
                transition_matrix[i, j] = state_stickiness
            if j == i+1:
                transition_matrix[i, j] = 1.-state_stickiness
            if i == num_latent_states-1 and j == 0:
                transition_matrix[i, j] = 1.-state_stickiness
    transition_matrix /= transition_matrix.sum(1, keepdims=True)

    # get initial state distrib as left eig vec
    e_vals, e_vecs = sp.linalg.eig(transition_matrix, left=True, right=False)
    one_eig_idx = np.argmin(np.abs(np.real(e_vals)-1.))
    assert np.imag(e_vals[one_eig_idx]) == 0
    ev_one = np.real(e_vecs[:, one_eig_idx])
    init_state_probs = ev_one / ev_one.sum()

    # create mean modulation parameters for different latent states
    # that are sampled far away from each other on an n-sphere
    means = samp_dist_npoints(num_comp, num_latent_states)

    # create variances such that there isnt too much overlap
    # also ensure good condition number
    dists = means.repeat(num_latent_states, 0)-np.tile(
        means, (num_latent_states, 1))
    dists = np.sqrt((dists**2).sum(1))
    min_dist = np.min(dists[dists.nonzero()])
    cond_num_best = 1e+9
    for i in range(1000):
        # idea that in worst case only 5% of distrib A is under
        # 95% of distrib B and vice versa
        sigmasq = np.random.uniform(0.001, (min_dist/(2*1.96))**2,
                                    size=(num_latent_states, num_comp))
        # write in natural parameter form
        natural_params_linear = means / sigmasq
        natural_params_square = -1. / sigmasq
        natural_params = np.zeros(shape=(num_latent_states, 2 * num_comp))
        natural_params[:, 0::2] = natural_params_linear
        natural_params[:, 1::2] = natural_params_square
        inv_nat_params = np.linalg.pinv(natural_params)
        cond_num = np.linalg.norm(
            natural_params)*np.linalg.norm(inv_nat_params)
        if cond_num < cond_num_best:
            cond_num_best = cond_num
            sigmasq_best = sigmasq.copy()
    sigmasq = sigmasq_best
    D = np.zeros(shape=(num_latent_states, num_comp, num_comp))
    for k in range(num_latent_states):
        D[k] = np.diag(sigmasq[k])

    # create latent state sequence
    state_sequence = np.zeros(sequence_length, dtype=np.int)
    for i in range(sequence_length):
        if i == 0:
            m_draw = np.random.multinomial(1, pvals=init_state_probs)
        else:
            m_draw = np.random.multinomial(
                1, pvals=transition_matrix[state_sequence[i-1], :])
        state_sequence[i] = np.argmax(m_draw)

    # generate HMM data for independent components (sources)
    source_data = np.zeros((sequence_length, num_comp))
    for t in range(sequence_length):
        k = state_sequence[t]
        source_data[t] = np.random.multivariate_normal(means[k], D[k])
    return source_data, state_sequence, means, D, transition_matrix

# gen_data(4, 5, 10, state_stickiness=0.99, radius=2, random_seed=0)