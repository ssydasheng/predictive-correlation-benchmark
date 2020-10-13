import tensorflow as tf
from tensorflow.python.ops.parallel_for import jacobian
import zhusuan as zs
import numpy as np
import core.gpflowSlim as gfs


from utils.common_utils import reuse_variables


def get_gpnet(name):
    if name == 'rf':
        return build_rf_expansion
    raise NameError('Not a supported name for posterior')


def build_rf_expansion(scope, layer_sizes, kernel, mvn=True, fix_freq=False,
                       residual=False, fix_ls=False, activation=tf.tanh, **kwargs):
    assert len(layer_sizes)==2
    x_dim, n_units = layer_sizes[0], layer_sizes[1]
    fixed_freq = np.random.normal(size=(x_dim, n_units))
    kernel, ls = kernel
    @reuse_variables(scope)
    def rf_expansion(x, x2=None, full_cov=True):
        # x: [batch_size, x_dim]
        if x2 is not None:
            assert full_cov
            assert not mvn
        input_ = tf.concat([x, x2], axis=0) if x2 is not None else x
        h = tf.to_double(input_)
        # freq: [x_dim, n_units]
        if fix_freq:
            freq = fixed_freq
        else:
            freq = tf.get_variable(
                "freq", dtype=tf.float64,
                initializer=tf.constant(fixed_freq, dtype=tf.float64))
        if fix_ls:
            lengthscales = ls  # kernel.lengthscales
        else:
            lengthscales = tf.get_variable(
                "ls", dtype=tf.float64, initializer=ls)  # kernel.lengthscales)
        # h: [batch_size, n_units]
        h = tf.matmul(h, freq * 1. / lengthscales[..., None])
        # h: [batch_size, 2 * n_units]
        h = tf.sqrt(kernel.variance) / np.sqrt(n_units) * tf.concat(
            [tf.cos(h), tf.sin(h)], -1)
        w_cov_raw = tf.get_variable(
            "w_cov", dtype=tf.float64,
            initializer=tf.eye(2 * n_units, dtype=tf.float64))
        w_cov_tril = tf.matrix_set_diag(
            tf.matrix_band_part(w_cov_raw, -1, 0),
            tf.nn.softplus(tf.matrix_diag_part(w_cov_raw)))
        # f_mean: [batch_size]
        f_mean = tf.squeeze(tf.layers.dense(h, 1, activation=None), -1)
        # f_mean_res: [batch_size]
        if residual:
            h_res = tf.layers.dense(input_, n_units, activation=activation)
            f_mean_res = tf.squeeze(tf.layers.dense(h_res, 1), -1)
            f_mean += f_mean_res
        # f_cov_half: [batch_size, 2 * n_units]
        f_cov_half = tf.matmul(h, w_cov_tril)
        if full_cov:
            if x2 is not None:
                f_cov = tf.matmul(f_cov_half[:tf.shape(x)[0]], f_cov_half[tf.shape(x)[0]:], transpose_b=True)
                return f_mean, f_cov
            # f_cov: [batch_size, batch_size]
            f_cov = tf.matmul(f_cov_half, f_cov_half, transpose_b=True)
            f_cov = f_cov + tf.eye(tf.shape(f_cov)[0], dtype=tf.float64) * \
                            gfs.settings.jitter
            if mvn:
                f_cov = tf.Print(f_cov, [f_cov], message='Full cov=')
                f_cov_tril = tf.cholesky(f_cov)
                f_dist = zs.distributions.MultivariateNormalCholesky(
                    f_mean, f_cov_tril)
                return f_dist
            else:
                return f_mean, f_cov
        else:
            # f_cov_diag: [batch_size]
            f_var = tf.reduce_sum(tf.square(f_cov_half), axis=-1)
            f_var = f_var + gfs.settings.jitter
            return f_mean, f_var
    return rf_expansion
