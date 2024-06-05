
import jax.numpy as jnp
import jax.scipy as jscipy

from jax import jit, vmap, jacfwd
from jax import ops, lax
from models import mlp
from functools import partial

func_estimator = mlp


@jit
def J_logllh_contri(params, input_data):

    J = jacfwd(func_estimator, argnums=1)
    data_J = vmap(partial(J, params))(input_data)
    _, data_logdetJ = jnp.linalg.slogdet(data_J)
    return data_logdetJ


@jit
def emission_llh(params, input_data, mu_est, D_est):

    # estimate the inverse mixing function
    s_est = func_estimator(params, input_data)

    # calculate emission probabilities using current parameters
    T = input_data.shape[0]
    K = mu_est.shape[0]
    logp_x_exc_J = jnp.zeros(shape=(T, K))
    for k in range(K):
        lpx_per_k = jscipy.stats.multivariate_normal.logpdf(s_est, mu_est[k], D_est[k])
        # logp_x_exc_J = ops.index_update(logp_x_exc_J, ops.index[:, k], lpx_per_k)
        logp_x_exc_J = logp_x_exc_J.at[:, k].set(lpx_per_k)
    logp_J = J_logllh_contri(params, input_data)
    logp_x = logp_x_exc_J + logp_J.reshape(-1, 1)
    return logp_x, logp_x_exc_J, logp_J, s_est


@jit
def mbatch_emission_llh(params, input_data, mu_est, D_est):

    return vmap(emission_llh, (None, 0, None, None),
                (0, 0, 0, 0))(params, input_data, mu_est, D_est)


@jit
def forward_backward_algo(logp_x, transition_matrix, init_probs):

    # set T and K
    T, K = logp_x.shape

    # transform into probabilities
    x_probs = jnp.exp(logp_x)

    # set up transition parameters
    A_est_ = transition_matrix
    pi_est_ = init_probs

    # define forward pass
    def forward_pass(t, fwd_msgs_and_scalers):
        scaled_fwd_msgs, scalers = fwd_msgs_and_scalers
        alpha = x_probs[t]*jnp.matmul(A_est_.T, scaled_fwd_msgs[t-1])
        # scaled_fwd_msgs = ops.index_update(scaled_fwd_msgs, t, alpha / alpha.sum())
        scaled_fwd_msgs = scaled_fwd_msgs.at[t].set(alpha / alpha.sum())
        # scalers = ops.index_update(scalers, t, alpha.sum())
        scalers = scalers.at[t].set(alpha.sum())
        fwd_msgs_and_scalers = (scaled_fwd_msgs, scalers)
        return fwd_msgs_and_scalers

    # initialize forward pass
    scalers = jnp.zeros(T)
    scaled_fwd_msgs = jnp.zeros(shape=(T, K))
    alpha = x_probs[0]*pi_est_
    # scaled_fwd_msgs = ops.index_update(scaled_fwd_msgs, 0, alpha/alpha.sum())
    scaled_fwd_msgs = scaled_fwd_msgs.at[0].set(alpha/alpha.sum())
    # scalers = ops.index_update(scalers, 0, alpha.sum())
    scalers = scalers.at[0].set(alpha.sum())
    fwd_msgs_and_scalers = (scaled_fwd_msgs, scalers)

    # note loop start from 1 since 0 was initialize
    scaled_fwd_msgs, scalers = lax.fori_loop(1, T, forward_pass,
                                             fwd_msgs_and_scalers)

    # define backward pass
    def backward_pass(t, scaled_bck_msgs):
        beta = jnp.matmul(A_est_, x_probs[-t]
                          * scaled_bck_msgs[-t]) / scalers[-t]
        # scaled_bck_msgs = ops.index_update(scaled_bck_msgs, ops.index[-(t+1)], beta)
        scaled_bck_msgs = scaled_bck_msgs.at[-(t+1)].set(beta)
        return scaled_bck_msgs

    # initialize backward pass
    scaled_bck_msgs = jnp.zeros(shape=(T, K))
    beta = jnp.ones(K)
    # scaled_bck_msgs = ops.index_update(scaled_bck_msgs, ops.index[-1], beta)
    scaled_bck_msgs = scaled_bck_msgs.at[-1].set(beta)

    # run backward pass
    scaled_bck_msgs = lax.fori_loop(1, T, backward_pass,
                                    scaled_bck_msgs)

    # calculate posteriors i.e. e-step
    marg_posteriors = scaled_fwd_msgs*scaled_bck_msgs
    pw_posteriors = jnp.zeros(shape=(T-1, K, K))

    def calc_pw_posteriors(t, pw_posteriors):
        pwm = jnp.dot(scaled_fwd_msgs[t].reshape(-1, 1),
                      (scaled_bck_msgs[t+1] * x_probs[t+1]).reshape(1, -1))
        pwm = pwm*A_est_ / scalers[t+1]
        # return ops.index_update(pw_posteriors, ops.index[t, :, :], pwm)
        return pw_posteriors.at[t, :, :].set(pwm)

    pw_posteriors = lax.fori_loop(0, T-1,
                                  calc_pw_posteriors,
                                  pw_posteriors)

    # to avoid numerical precision issues
    eps = 1e-30
    marg_posteriors = jnp.clip(marg_posteriors, a_min=eps)
    pw_posteriors = jnp.clip(pw_posteriors, a_min=eps)
    return marg_posteriors, pw_posteriors, scalers


@jit
def mbatch_fwd_bwd(logp_x, transition_matrix, init_probs):

    return vmap(forward_backward_algo, (0, None, None),
                (0, 0, 0))(logp_x, transition_matrix, init_probs)


@jit
def mbatch_m_step(s_est, marg_posteriors, pw_posteriors):

    # set dimensions
    N = s_est.shape[-1]
    K = marg_posteriors.shape[-1]
    # update mean parameters
    mu_est = (jnp.expand_dims(s_est, -1)
              * jnp.expand_dims(marg_posteriors, -2)).sum((0, 1))
    mu_est /= marg_posteriors.sum((0, 1)).reshape(1, -1)
    mu_est = mu_est.T

    # update covariance matrices for all latent states and weigh across minib.
    dist_to_mu = s_est[:, jnp.newaxis, :, :]-mu_est[jnp.newaxis, :,
                                                    jnp.newaxis, :]
    cov_est = jnp.einsum('bktn, bktm->bktnm', dist_to_mu, dist_to_mu)
    wgt_cov_est = (cov_est*jnp.transpose(
        marg_posteriors, (0, 2, 1))[:, :, :, jnp.newaxis,
                                    jnp.newaxis]).sum((0, 2))
    D_est = wgt_cov_est / marg_posteriors.sum((0, 1))[:, jnp.newaxis,
                                                      jnp.newaxis]

    # set lowerbound to avoid heywood cases
    eps = 1e-4
    D_est = jnp.clip(D_est, a_min=eps)
    D_est = D_est*jnp.eye(N).reshape(1, N, N)

    # update latent state transitions (notice the prior)
    hyperobs = 1 # i.e. a-1 ; a=2 where a is hyperprior or dirichlet
    expected_counts = pw_posteriors.sum((0, 1))
    A_est = (expected_counts + hyperobs) / (
        K*hyperobs + marg_posteriors.sum((0, 1)).reshape(-1, 1))

    # update initial state probabilities
    pi_est = marg_posteriors.mean(0)[0] + eps
    pi_est = pi_est/pi_est.sum()
    return mu_est, D_est, A_est, pi_est


#@jit
def viterbi(logp_x, transition_matrix, init_probs):

    # set up T and K
    T, K = logp_x.shape

    # set up transition parameters
    A_est_ = transition_matrix
    pi_est_ = init_probs

    # define forward pass
    def forward_pass(t, fwd_msgs_and_paths):
        fwd_msgs, best_paths = fwd_msgs_and_paths

        msg = logp_x[t]+jnp.max(jnp.log(A_est_)
                                + fwd_msgs[t-1].reshape(-1, 1), 0)
        max_prev_state = jnp.argmax(jnp.log(A_est_)
                                    + fwd_msgs[t-1].reshape(-1, 1), 0)
        fwd_msgs = fwd_msgs.at[t, :].set(msg)
        best_paths = best_paths.at[t-1].set(max_prev_state)
        fwd_msgs_and_paths = (fwd_msgs, best_paths)
        return fwd_msgs_and_paths

    # initialize forward pass
    fwd_msgs = jnp.zeros(shape=(T, K))
    best_paths = jnp.zeros(shape=(T, K), dtype=jnp.int32)
    msg = logp_x[0] + jnp.log(pi_est_)
    fwd_msgs = fwd_msgs.at[0].set(msg)

    fwd_msgs_and_paths = (fwd_msgs, best_paths)
    fwd_msgs, best_paths = lax.fori_loop(1, T, forward_pass,
                                         fwd_msgs_and_paths)

    # define backward pass
    def backward_pass(t, the_best_path):
        best_k = best_paths[-(t+1), the_best_path[-t]]
        the_best_path = the_best_path.at[-(t+1)].set(best_k)
        return the_best_path

    # initialize backward pass
    the_best_path = jnp.zeros(shape=(T,), dtype=jnp.int32)
    best_k = jnp.argmax(fwd_msgs[-1])
    the_best_path = the_best_path.at[-1].set(best_k)

    # run backward pass
    the_best_path = lax.fori_loop(1, T, backward_pass,
                                  the_best_path)
    return the_best_path


