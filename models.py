import pdb
import jax.numpy as jnp
import numpy as np

from jax import random as jrandom
from jax import nn as jnn

from utils import l2_norm, find_mat_thresh, Sm_LeakyRelu


def unif_invert_weights(key, in_dim, out_dim, w_cond_thresh,
                                  weight_range, bias_range):

    # ensure good condition number for weight matrix
    W_key, b_key = jrandom.split(key)
    if w_cond_thresh is None:
        W = jrandom.uniform(W_key, (in_dim, out_dim), minval=weight_range[0],
                            maxval=weight_range[1])
        W = np.tril(W)
        W = l2_norm(W, 1)
    else:
        cond_W = w_cond_thresh + 1
        while cond_W > w_cond_thresh:
            W_key, subkey = jrandom.split(W_key)
            W = jrandom.uniform(subkey, (in_dim, out_dim),
                                minval=weight_range[0],
                                maxval=weight_range[1])
            W = np.tril(W)
            W = l2_norm(W, 1)
            cond_W = np.linalg.cond(W)
    b = jrandom.uniform(b_key, (out_dim,), minval=bias_range[0],
                        maxval=bias_range[1])
    return W, b


def init_invert_mlp_params(key, dim, num_layers,
                               weight_range=[-1., 1.], bias_range=[0., 1.]):

    keys = jrandom.split(key, num_layers)
    ct = find_mat_thresh(dim, weight_range)
    return [unif_invert_weights(k, d_in, d_out,
                                          ct, weight_range, bias_range)
            for k, d_in, d_out in zip(keys, [dim]*num_layers,
                                      [dim]*num_layers)]





def invertible_mlp_fwd(params, x, slope=0.1):

    z = x
    for W, b in params[:-1]:
        W = np.triu(W)
        z = jnp.matmul(z, W)
        z = jnn.leaky_relu(z, slope)
    final_W, final_b = params[-1]
    final_W = np.triu(final_W)
    z = jnp.dot(z, final_W)
    return z

def invertible_mlp_inverse(params, x, lrelu_slope=0.1):

    z = x
    params_rev = params[::-1]
    final_W, final_b = params_rev[0]
    z = z - final_b
    z = jnp.dot(z, jnp.linalg.inv(final_W))
    for W, b in params_rev[1:]:
        z = jnn.leaky_relu(z, 1./lrelu_slope)
        z = z - b
        z = jnp.dot(z, jnp.linalg.inv(W))
    return z


def init_mlp_params(key, layer_sizes):

    keys = jrandom.split(key, len(layer_sizes))
    return [unif_invert_weights(k, m, n, None, [-1., 1.], [0., 0.1])
            for k, m, n in zip(keys, layer_sizes[:-1], layer_sizes[1:])]


def mlp(params, inputs, slope=0.1):

    activation = Sm_LeakyRelu(slope)
    z = inputs
    for W, b in params[:-1]:
        z = jnp.matmul(z, W)+b
        z = activation(z)
    final_W, final_b = params[-1]
    z = jnp.matmul(z, final_W) + final_b
    return z
