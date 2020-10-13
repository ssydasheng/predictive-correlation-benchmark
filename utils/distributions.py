import tensorflow as tf

def compute_mvg_kl_divergence(param1, param2, jitter=1e-8,
                              sqrt_u1=False, sqrt_v1=False, sqrt_u2=False, sqrt_v2=False,
                              lower_u1=False, lower_v1=False):
    mean1, u1_or_sqrt_u1, v1_or_sqrt_v1 = param1
    mean2, u2_or_sqrt_u2, v2_or_sqrt_v2 = param2
    n, m = tf.shape(mean1)[0], tf.shape(mean1)[1]

    jitter_u = tf.eye(n, dtype=tf.float64) * jitter
    jitter_v = tf.eye(m, dtype=tf.float64) * jitter

    if sqrt_u1 is False:
        u1 = u1_or_sqrt_u1 + jitter_u
        u1_tril = tf.cholesky(u1 + jitter_u)
        lower_u1 = True
    else:
        u1_tril = u1_or_sqrt_u1
        u1 = tf.matmul(u1_tril, u1_tril, transpose_b=True)
    if sqrt_v1 is False:
        v1 = v1_or_sqrt_v1 + jitter_v
        v1_tril = tf.cholesky(v1 + jitter_v)
        lower_v1 = True
    else:
        v1_tril = v1_or_sqrt_v1
        v1 = tf.matmul(v1_tril, v1_tril, transpose_b=True)
    if sqrt_u2 is False:
        u2 = u2_or_sqrt_u2 + jitter_u
        u2_tril = tf.cholesky(u2 + jitter_u)
    else:
        u2_tril = u2_or_sqrt_u2
        u2 = tf.matmul(u2_tril, u2_tril, transpose_b=True)
    if sqrt_v2 is False:
        v2 = v2_or_sqrt_v2 + jitter_v
        v2_tril = tf.cholesky(v2 + jitter_v)
    else:
        v2_tril = v2_or_sqrt_v2
        v2 = tf.matmul(v2_tril, v2_tril, transpose_b=True)

    n, m = tf.cast(n, mean1.dtype), tf.cast(m, mean1.dtype)
    u1_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u1_tril)))) * 2.
    v1_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v1_tril)))) * 2.
    logdet_1 = u1_logdet + v1_logdet

    u2_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u2_tril)))) * 2.
    v2_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v2_tril)))) * 2.
    logdet_2 = u2_logdet + v2_logdet

    logdet_difference = logdet_2 - logdet_1
    const = tf.cast(n * m, mean1.dtype)

    vec = mean1-mean2
    inverse_cov2_vec = tf.cholesky_solve(u2_tril, tf.transpose(tf.cholesky_solve(v2_tril, tf.transpose(vec))))
    mean_diff = tf.reduce_sum(vec * inverse_cov2_vec)

    trace = tf.trace(tf.cholesky_solve(v2_tril, v1)) * tf.trace(tf.cholesky_solve(u2_tril, u1))

    kl = 0.5 * (logdet_difference - const + trace + mean_diff)
    return kl