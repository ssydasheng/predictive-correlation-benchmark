import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def get_submatrices(cov, *args):
    n = len(args)
    return np.reshape(np.concatenate([cov[args[i1], args[i2]][:, None] for i2 in range(n) for i1 in range(n)],
        axis=1), [-1, len(args), len(args)])


def sort_corr(cov):
    diag = np.diagonal(cov)[None]
    corr = cov / (diag * diag.T) ** 0.5
    sorted_indices = np.argsort(corr, axis=1)  # [n, n], each value is index
    return sorted_indices[:, ::-1]


def get_indices(INDICES_TYPE, n_test, n_sub, N_SAMPLES, refer_cov):
    if INDICES_TYPE == 'random':
        return np.concatenate([np.random.choice(range(n_test), size=[n_sub], replace=False)[None]
                               for _ in range(N_SAMPLES)], axis=0).T
    elif INDICES_TYPE == 'topk':
        topk_refer_index = sort_corr(refer_cov)[:, :n_sub].T # [n_sub, n_test]
        return topk_refer_index
    elif INDICES_TYPE == 'topk_abs':
        topk_refer_index = sort_corr(np.abs(refer_cov))[:, :n_sub].T # [n_sub, n_test]
        return topk_refer_index
    else:
        raise NotImplementedError


def build_graph(n_sub):
    mean = tf.placeholder(tf.float64, shape=[None, n_sub])
    y = tf.placeholder(tf.float64, shape=[None, n_sub])
    cov = tf.placeholder(tf.float64, shape=[None, n_sub, n_sub])
    dist = tfp.distributions.MultivariateNormalFullCovariance(mean, cov)
    log_prob = tf.reduce_mean(dist.log_prob(y))
    return log_prob, mean, y, cov
