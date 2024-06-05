import pdb
import time
import itertools

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from jax import value_and_grad, jit
from jax import lax
from jax import ops
from jax.experimental import optimizers
from jax.numpy import ndarray

from models import init_mlp_params
from functions import mbatch_emission_llh, emission_llh
from functions import mbatch_fwd_bwd, mbatch_m_step
from functions import forward_backward_algo, viterbi
from utils import match_cor, cluster_accuracy



def train(data_dict, train_dict, seed_dict, results_dict):

    # unpack data
    x = data_dict['x_data']
    s_true = data_dict['s_data']
    state_seq = data_dict['state_seq']

    # set data dimensions
    N = x.shape[1]
    T = x.shape[0]
    K = len(np.unique(state_seq))

    # unpack training variables
    mix_depth = train_dict['mix_depth']
    hidden_size = train_dict['hidden_size']
    learning_rate = train_dict['learning_rate']
    num_epochs = train_dict['num_epochs']
    subseq_len = train_dict['subseq_len']
    minib_size = train_dict['minib_size']
    decay_rate = train_dict['decay_rate']
    decay_steps = train_dict['decay_steps']

    print("Training with N={n}, T={t}, K={k}\t"
          "mix_depth={md}".format(n=N, t=T, k=K, md=mix_depth))

    # initialize parameters for mlp function approximator
    key = jrandom.PRNGKey(seed_dict['est_mlp_seed'])
    layer_sizes = [N]+[hidden_size]*(mix_depth-1)+[N]
    mlp_params = init_mlp_params(key, layer_sizes)

    # initialize parameters for estimating distribution parameters
    np.random.seed(seed_dict['est_distrib_seed'])
    mu_est = np.random.uniform(-5., 5., size=(K, N))
    var_est = np.random.uniform(1., 2., size=(K, N))
    D_est = np.zeros(shape=(K, N, N))
    for k in range(K):
        D_est[k] = np.diag(var_est[k])

    # initialize transition parameter estimates
    A_est = np.eye(K) + 0.05
    A_est = A_est / A_est.sum(1, keepdims=True)
    pi_est = A_est.sum(0)/A_est.sum()

    # set up optimizer
    schedule = optimizers.exponential_decay(learning_rate,
                                            decay_steps=decay_steps,
                                            decay_rate=decay_rate)
    opt_init, opt_update, get_params = optimizers.adam(schedule)

    # set up loss function and training step
    @jit
    def calc_loss(params, input_data, marginal_posteriors,
                  mu_est, D_est, num_subseqs):
        """Calculates the loss for gradient M-step for function estimator.
        """
        lp_x, lp_x_exc_J, lp_J, _ = mbatch_emission_llh(params,
                                                               input_data,
                                                               mu_est, D_est)
        expected_lp_x = jnp.sum(marginal_posteriors*lp_x, -1)
        # note correction for bias below
        return -expected_lp_x.mean()*num_subseqs

    @jit
    def training_step(iter_num, input_data, marginal_posteriors,
                      mu_est, D_est, opt_state, num_subseqs):
        """Performs gradient m-step on the function estimator
               MLP parameters.
        """
        params = get_params(opt_state)
        loss, g = value_and_grad(calc_loss, argnums=0)(
            params, input_data,
            lax.stop_gradient(marginal_posteriors),
            mu_est, D_est, num_subseqs
        )
        return loss, opt_update(iter_num, g, opt_state)

    # function to load subsequence data for minibatches
    @jit
    def get_subseq_data(orig_data, subseq_array_to_fill):
        """Collects all sub-sequences into an array.
        """
        subseq_data = subseq_array_to_fill
        num_subseqs = subseq_data.shape[0]
        subseq_len = subseq_data.shape[1]

        def body_fun(i, subseq_data):
            """Function to loop over.
            """
            subseq_i = lax.dynamic_slice_in_dim(orig_data, i, subseq_len)
            # subseq_data = ops.index_update(subseq_data, ops.index[i, :, :], subseq_i)
            subseq_data = subseq_data.at[i, :, :].set(subseq_i)
            return subseq_data
        return lax.fori_loop(0, num_subseqs, body_fun, subseq_data)

    # set up minibatch training
    num_subseqs = T-subseq_len+1
    assert num_subseqs >= minib_size
    num_full_minibs, remainder = divmod(num_subseqs, minib_size)
    num_minibs = num_full_minibs + bool(remainder)
    sub_data_holder = jnp.zeros((num_subseqs, subseq_len, N))
    sub_data = get_subseq_data(x, sub_data_holder)
    print("T: {t}\t"
          "subseq_len: {slen}\t"
          "minibatch size: {mbs}\t"
          "num minibatches: {nbs}".format(
              t=T, slen=subseq_len, mbs=minib_size, nbs=num_minibs))

    # initialize and train
    best_logl = -np.inf
    itercount = itertools.count()
    opt_state = opt_init(mlp_params)
    all_subseqs_idx = np.arange(num_subseqs)
    for epoch in range(num_epochs):
        tic = time.time()
        # shuffle subseqs for added stochasticity
        np.random.shuffle(all_subseqs_idx)
        sub_data = sub_data.copy()[all_subseqs_idx]
        # train over minibatches
        for batch in range(num_minibs):
            # select sub-sequence for current minibatch
            batch_data = sub_data[batch*minib_size:(batch+1)*minib_size]

            # calculate emission likelihood using most recent parameters
            params = get_params(opt_state)
            logp_x, logp_x_exc_J, lpj, s_est = mbatch_emission_llh(
                params, batch_data, mu_est, D_est
            )

            # forward-backward algorithm
            marg_posteriors, pw_posteriors, scalers = mbatch_fwd_bwd(
                logp_x, A_est, pi_est
            )

            # exact M-step for mean and variance
            mu_est, D_est, A_est, pi_est = mbatch_m_step(s_est,
                                                         marg_posteriors,
                                                         pw_posteriors)

            # SGD for mlp parameters
            loss, opt_state = training_step(next(itercount), batch_data,
                                            marg_posteriors, mu_est, D_est,
                                            opt_state, num_subseqs)

        # gather full data after each epoch for evaluation
        params_latest = get_params(opt_state)
        logp_x_all, _, _, s_est_all = emission_llh(
               params_latest, x, mu_est, D_est
        )

        _, _, scalers = forward_backward_algo(
                logp_x_all, A_est, pi_est
            )
        logl_all = np.log(scalers).sum()

        # viterbi to estimate state prediction
        est_seq = viterbi(logp_x_all, A_est, pi_est)
        cluster_acc = cluster_accuracy(np.array(est_seq), np.array(state_seq))

        # evaluate correlation of estimated and true independent components
        mean_abs_corr, s_est_sorted, sort_idx = match_cor(
            np.array(s_est_all), np.array(s_true)
        )

        # save results
        if logl_all > best_logl:
            best_logl = logl_all
            best_logl_corr = mean_abs_corr
            best_logl_acc = cluster_acc
            results_dict['results'].append({'best_logl': best_logl,
                                            'best_logl_corr': mean_abs_corr,
                                            'best_logl_acc': cluster_acc})

        results_dict['results'].append({'epoch': epoch,
                                        'logl': logl_all,
                                        'corr': mean_abs_corr,
                                        'acc': cluster_acc})
        # print them
        print("Epoch: [{0}/{1}]\t"
              "LogL: {logl:.2f}\t"
              "mean corr between s and s_est {corr:.2f}\t"
              "acc {acc:.2f}\t"
              "elapsed {time:.2f}".format(
                  epoch, num_epochs, logl=logl_all, corr=mean_abs_corr,
                  acc=cluster_acc, time=time.time()-tic))

    # pack data into tuples
    results_dict['results'].append({'best_logl': best_logl,
                                    'best_logl_corr': best_logl_corr,
                                    'best_logl_acc': best_logl_acc})
    est_params = (mu_est, D_est, A_est, est_seq)
    return s_est_all, sort_idx, results_dict, est_params
