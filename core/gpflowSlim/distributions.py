import tensorflow as tf
import tensorflow_probability as tfp

from . import settings


class GaussianDistribution:

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def var(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

class UnivariateGaussian(GaussianDistribution):
    def __init__(self, mean, var):
        self._mean = mean
        self._var = var

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return  self._var

    def sample(self):
        eps = tf.random_normal(tf.shape(self.mean), dtype=self.mean.dtype)
        return self.mean + eps * tf.sqrt(self.var)


class MultivariateGaussian(GaussianDistribution):

    def __init__(self, mean, var):
        self._mean = mean
        self._cov = var

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._cov

    def sample(self):
        dist = tfp.distributions.MultivariateNormalFullCovariance(self.mean, self.var)
        return dist.sample()


class MatrixVariateGaussian(GaussianDistribution):

    def __init__(self, mean, cov_u=None, cov_v=None, cov_u_sqrt=None, cov_v_sqrt=None):
        assert cov_u is not None or cov_u_sqrt is not None, 'cov and cov_sqrt cannot be None at the same time'
        assert cov_u is None or cov_u_sqrt is None, 'one of cov and cov_sqrt must be None'
        assert cov_v is not None or cov_v_sqrt is not None, 'cov and cov_sqrt cannot be None at the same time'
        assert cov_v is None or cov_v_sqrt is None, 'one of cov and cov_sqrt must be None'

        self._mean = mean
        n, m  = tf.shape(mean)[-2], tf.shape(mean)[-1]

        if cov_u is not None:
            self._cov_u = cov_u
            self._cov_u_sqrt = tf.cholesky(cov_u + settings.jitter * tf.eye(n, dtype=settings.tf_float))
        else:
            self._cov_u = tf.matmul(cov_u_sqrt, cov_u_sqrt, transpose_b=True)
            self._cov_u_sqrt = cov_u_sqrt

        if cov_v is not None:
            self._cov_v = cov_v
            self._cov_v_sqrt = tf.cholesky(cov_v + settings.jitter * tf.eye(m, dtype=settings.tf_float))
        else:
            self._cov_v = tf.matmul(cov_v_sqrt, cov_v_sqrt, transpose_b=True)
            self._cov_v_sqrt = cov_v_sqrt

        if len(mean.get_shape().as_list()) == 3:
            s = tf.shape(mean)[0]
            if len(self._cov_v.get_shape().as_list()) == 2:
                self._cov_v = tf.tile(self._cov_v[None, ...], [s, 1, 1])
                self._cov_v_sqrt = tf.tile(self._cov_v_sqrt[None, ...], [s, 1, 1])

    @property
    def mean(self):
        return self._mean

    @property
    def cov_u_sqrt(self):
        return self._cov_u_sqrt

    @property
    def cov_v_sqrt(self):
        return self._cov_v_sqrt

    @property
    def cov_u(self):
        return self._cov_u

    @property
    def cov_v(self):
        return self._cov_v

    @property
    def var(self):
        return tf.matmul(tf.matrix_diag_part(self._cov_u)[..., None],
                         tf.matrix_diag_part(self._cov_v)[..., None, :])

    def sample(self):
        eps = tf.random.normal(tf.shape(self.mean), dtype=settings.tf_float)
        return self.mean + tf.matmul(tf.matmul(self._cov_u_sqrt, eps), self._cov_v_sqrt, transpose_b=True)


class MatrixVariateDiagGaussian(MatrixVariateGaussian):
    """
        A gaussian distribution of covariance (cov_v \otimes var_u**0.5) diag (cov_v \otimes var_u**0.5)^T

        :param mean  : [n, d]
        :param cov_v : [d, d]
        :param var_u : [n]
        :param diag  : [n, d]
        """
    def __init__(self, mean, diag, cov_u_sqrt, cov_v=None, cov_v_sqrt=None):
        super(MatrixVariateDiagGaussian, self).__init__(mean, cov_v=cov_v,
                                                        cov_u_sqrt=cov_u_sqrt, cov_v_sqrt=cov_v_sqrt)
        self._diag = diag

    @property
    def diag(self):
        return self._diag

    @property
    def var(self):
        return tf.matmul(self.cov_u_sqrt**2., tf.matmul(self.diag, self.cov_v_sqrt**2., transpose_b=True))

    def sample(self):
        # TODO: here different points share the same random number
        eps = tf.random.normal(tf.shape(self.diag), dtype=settings.tf_float)
        eps = eps * tf.sqrt(self.diag)
        return self.mean + tf.matmul(tf.matmul(self._cov_u_sqrt, eps), self._cov_v_sqrt, transpose_b=True)


class RowIndependentDiagGaussian_1(MatrixVariateDiagGaussian):
    """
        A gaussian distribution of covariance (cov_v \otimes var_u**0.5) diag (cov_v \otimes var_u**0.5)^T

        :param mean  : [n, d]
        :param cov_v : [d, d]
        :param cov_u_sqrt : [n, m]
        :param diag  : [m, d]
        """
    def sample(self): #TODO: to make it memory efficient
        n, m, d = tf.shape(self.cov_u_sqrt)[0], tf.shape(self.cov_u_sqrt)[1], tf.shape(self.mean)[-1]
        eps = tf.random.normal([n, m, d], dtype=settings.tf_float)
        eps = eps * tf.sqrt(self.diag)
        return self.mean + tf.matmul(tf.squeeze(tf.matmul(self._cov_u_sqrt[:, None], eps), 1),
                                     self._cov_v_sqrt, transpose_b=True)

class RowIndependentDiagGaussian_2(MatrixVariateDiagGaussian):
    def sample(self): #TODO: to make it memory efficient
        n, m, d = tf.shape(self.cov_u_sqrt)[0], tf.shape(self.cov_u_sqrt)[1], tf.shape(self.mean)[-1]

        # std = tf.sqrt(tf.reduce_sum(self._cov_u_sqrt[..., None]**2. * self.diag[None, ...], 1))
        std = tf.sqrt(tf.matmul(self._cov_u_sqrt**2., self.diag))
        eps = tf.random_normal([n, d], dtype=settings.tf_float)
        return self.mean + tf.matmul(std * eps, self._cov_v_sqrt, transpose_b=True)


class RowIndependentMVG(GaussianDistribution):

    def __init__(self, mean, var_u, cov_v=None, cov_v_sqrt=None):
        assert cov_v is not None or cov_v_sqrt is not None, 'cov and cov_sqrt cannot be None at the same time'
        assert cov_v is None or cov_v_sqrt is None, 'one of cov and cov_sqrt must be None'

        self._mean = mean
        n, m  = tf.shape(mean)[-2], tf.shape(mean)[-1]
        self._var_u = var_u

        if cov_v is not None:
            self._cov_v = cov_v
            self._cov_v_sqrt = tf.cholesky(cov_v + settings.jitter * tf.eye(m, dtype=settings.tf_float))
        else:
            self._cov_v = tf.matmul(cov_v_sqrt, cov_v_sqrt, transpose_b=True)
            self._cov_v_sqrt = cov_v_sqrt

        if len(mean.get_shape().as_list()) == 3:
            s = tf.shape(mean)[0]
            if len(self._cov_v.get_shape().as_list()) == 2:
                self._cov_v = tf.tile(self._cov_v[None, ...], [s, 1, 1])
                self._cov_v_sqrt = tf.tile(self._cov_v_sqrt[None, ...], [s, 1, 1])

    @property
    def mean(self):
        return self._mean

    @property
    def cov_v_sqrt(self):
        return self._cov_v_sqrt

    @property
    def var_u(self):
        return self._var_u

    @property
    def cov_v(self):
        return self._cov_v

    @property
    def var(self):
        return tf.matmul(self.var_u[..., None],
                         tf.matrix_diag_part(self._cov_v)[..., None, :])

    def sample(self):
        eps = tf.random.normal(tf.shape(self.mean), dtype=settings.tf_float)
        return self.mean + tf.sqrt(self.var_u[..., None]) * tf.matmul(eps, self._cov_v_sqrt, transpose_b=True)


class SumGaussian(GaussianDistribution):
    def __init__(self, gausses):
        self._gausses = gausses

    @property
    def mean(self):
        return tf.add_n([g.mean for g in self._gausses])

    @property
    def var(self):
        return tf.add_n([g.var for g in self._gausses])

    def sample(self):
        return tf.add_n([g.sample() for g in self._gausses])


def compute_mvg_kl_divergence(param1, param2, jitter=1e-8,
                              sqrt_u1=False, sqrt_v1=False, sqrt_u2=False, sqrt_v2=False):
    mean1, u1_or_sqrt_u1, v1_or_sqrt_v1 = param1
    mean2, u2_or_sqrt_u2, v2_or_sqrt_v2 = param2
    n, m = tf.shape(mean1)[0], tf.shape(mean1)[1]

    jitter_u = tf.eye(n, dtype=tf.float64) * jitter
    jitter_v = tf.eye(m, dtype=tf.float64) * jitter

    if sqrt_u1 is False:
        u1 = u1_or_sqrt_u1 + jitter_u
        u1_tril = tf.cholesky(u1 + jitter_u)
    else:
        u1_tril = u1_or_sqrt_u1
        u1 = tf.matmul(u1_tril, u1_tril, transpose_b=True)
    if sqrt_v1 is False:
        v1 = v1_or_sqrt_v1 + jitter_v
        v1_tril = tf.cholesky(v1 + jitter_v)
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

    n, m = tf.to_double(n), tf.to_double(m)
    u1_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u1_tril)))) * 2.
    v1_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v1_tril)))) * 2.
    logdet_1 = u1_logdet + v1_logdet

    u2_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u2_tril)))) * 2.
    v2_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v2_tril)))) * 2.
    logdet_2 = u2_logdet + v2_logdet

    logdet_difference = logdet_2 - logdet_1
    const = tf.to_double(n * m)

    vec = mean1-mean2
    inverse_cov2_vec = tf.cholesky_solve(u2_tril, tf.transpose(tf.cholesky_solve(v2_tril, tf.transpose(vec))))
    mean_diff = tf.reduce_sum(vec * inverse_cov2_vec)

    trace = tf.trace(tf.cholesky_solve(v2_tril, v1)) * tf.trace(tf.cholesky_solve(u2_tril, u1))

    kl = 0.5 * (logdet_difference - const + trace + mean_diff)
    return kl


def compute_summvg_mvg_kl_divergence(
        param11, param12, param2, jitter=1e-8,
        sqrt_u11=False, sqrt_v11=False, sqrt_u12=False, sqrt_v12=False, sqrt_u2=False, sqrt_v2=False):
    # q = N(m11+m12, u11 x v11 + u12 x v12)
    # p = N(m2, u2 x v2)

    mean11, u11_or_sqrt_u11, v11_or_sqrt_v11 = param11
    mean12, u12_or_sqrt_u12, v12_or_sqrt_v12 = param12
    mean2, u2_or_sqrt_u2, v2_or_sqrt_v2 = param2
    n, m = tf.shape(mean11)[0], tf.shape(mean11)[1]

    jitter_u = tf.eye(n, dtype=tf.float64) * jitter
    jitter_v = tf.eye(m, dtype=tf.float64) * jitter

    if sqrt_u11 is False:
        u11 = u11_or_sqrt_u11 + jitter_u
        u11_tril = tf.cholesky(u11 + jitter_u)
    else:
        u11_tril = u11_or_sqrt_u11
        u11 = tf.matmul(u11_tril, u11_tril, transpose_b=True)
    if sqrt_v11 is False:
        v11 = v11_or_sqrt_v11 + jitter_v
        v11_tril = tf.cholesky(v11 + jitter_v)
    else:
        v11_tril = v11_or_sqrt_v11
        v11 = tf.matmul(v11_tril, v11_tril, transpose_b=True)
    if sqrt_u12 is False:
        u12 = u12_or_sqrt_u12 + jitter_u
        u12_tril = tf.cholesky(u12 + jitter_u)
    else:
        u12_tril = u12_or_sqrt_u12
        u12 = tf.matmul(u12_tril, u12_tril, transpose_b=True)
    if sqrt_v12 is False:
        v12 = v12_or_sqrt_v12 + jitter_v
        v12_tril = tf.cholesky(v12 + jitter_v)
    else:
        v12_tril = v12_or_sqrt_v12
        v12 = tf.matmul(v12_tril, v12_tril, transpose_b=True)
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

    n, m = tf.to_double(n), tf.to_double(m)

    u12_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u12_tril)))) * 2.
    v12_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v12_tril)))) * 2.
    logdet_12 = u12_logdet + v12_logdet

    tmpu = tf.matrix_triangular_solve(u12_tril, u11)
    inv12_u11 = tf.transpose(tf.matrix_triangular_solve(u12_tril, tf.transpose(tmpu)))
    tmpv = tf.matrix_triangular_solve(v12_tril, v11)
    inv12_v11 = tf.transpose(tf.matrix_triangular_solve(v12_tril, tf.transpose(tmpv)))

    u_eig12 = tf.linalg.eigvalsh(inv12_u11)
    v_eig12 = tf.linalg.eigvalsh(inv12_v11)
    logdet_prod2 = tf.reduce_sum(tf.log(u_eig12[:, None] * v_eig12[None, :] + 1.))
    logdet_1 = logdet_12 + logdet_prod2

    u2_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u2_tril)))) * 2.
    v2_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v2_tril)))) * 2.
    logdet_2 = u2_logdet + v2_logdet

    logdet_difference = logdet_2 - logdet_1
    const = tf.to_double(n * m)

    vec = mean11 + mean12 - mean2
    inverse_cov2_vec = tf.cholesky_solve(u2_tril, tf.transpose(tf.cholesky_solve(v2_tril, tf.transpose(vec))))
    mean_diff = tf.reduce_sum(vec * inverse_cov2_vec)

    trace1 = tf.trace(tf.cholesky_solve(v2_tril, v11)) * tf.trace(tf.cholesky_solve(u2_tril, u11))
    trace2 = tf.trace(tf.cholesky_solve(v2_tril, v12)) * tf.trace(tf.cholesky_solve(u2_tril, u12))
    trace = trace1 + trace2

    kl = 0.5 * (logdet_difference - const + trace + mean_diff)
    return kl


def compute_prec_summvg_mvg_kl_divergence(
        param11, param12, param2, jitter=1e-8,
        sqrt_u11=False, sqrt_v11=False, sqrt_u12=False, sqrt_v12=False, sqrt_u2=False, sqrt_v2=False,
        lower_v12=True, lower_u12=True, eigh_inv12_u11=None, eigh_inv12_v11=None):
    # q = N(m11+m12, (v11 x u11 + v12 x u12)^{-1})
    # p = N(m2, v2 x u2)

    mean11, u11_or_sqrt_u11, v11_or_sqrt_v11 = param11
    mean12, u12_or_sqrt_u12, v12_or_sqrt_v12 = param12
    mean2, u2_or_sqrt_u2, v2_or_sqrt_v2 = param2
    n, m = tf.shape(mean11)[0], tf.shape(mean11)[1]

    jitter_u = tf.eye(n, dtype=tf.float64) * jitter
    jitter_v = tf.eye(m, dtype=tf.float64) * jitter

    if sqrt_u11 is False:
        u11 = u11_or_sqrt_u11 + jitter_u
        u11_tril = tf.cholesky(u11 + jitter_u)
    else:
        u11_tril = u11_or_sqrt_u11
        u11 = tf.matmul(u11_tril, u11_tril, transpose_b=True)
    if sqrt_v11 is False:
        v11 = v11_or_sqrt_v11 + jitter_v
        v11_tril = tf.cholesky(v11 + jitter_v)
    else:
        v11_tril = v11_or_sqrt_v11
        v11 = tf.matmul(v11_tril, v11_tril, transpose_b=True)
    if sqrt_u12 is False:
        u12 = u12_or_sqrt_u12 + jitter_u
        u12_tril = tf.cholesky(u12 + jitter_u)
        lower_u12 = True
    else:
        u12_tril = u12_or_sqrt_u12
        u12 = tf.matmul(u12_tril, u12_tril, transpose_b=True)
    if sqrt_v12 is False:
        v12 = v12_or_sqrt_v12 + jitter_v
        v12_tril = tf.cholesky(v12 + jitter_v)
        lower_v12 = True
    else:
        v12_tril = v12_or_sqrt_v12
        v12 = tf.matmul(v12_tril, v12_tril, transpose_b=True)
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

    n, m = tf.to_double(n), tf.to_double(m)

    u12_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u12_tril)))) * 2.
    v12_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v12_tril)))) * 2.
    logdet_12 = u12_logdet + v12_logdet

    if eigh_inv12_u11 is None:
        tmpu = tf.matrix_triangular_solve(u12_tril, u11, lower=lower_u12)
        inv12_u11 = tf.transpose(tf.matrix_triangular_solve(u12_tril, tf.transpose(tmpu), lower=lower_u12))
        with tf.device('/cpu:0'): #TODO
            u_eigv12, u_eig_vec12 = tf.linalg.eigh(inv12_u11)
    else:
        u_eigv12, u_eig_vec12 = eigh_inv12_u11

    if eigh_inv12_v11 is None:
        tmpv = tf.matrix_triangular_solve(v12_tril, v11, lower=lower_v12)
        inv12_v11 = tf.transpose(tf.matrix_triangular_solve(v12_tril, tf.transpose(tmpv), lower=lower_v12))
        with tf.device('/cpu:0'):
            v_eigv12, v_eig_vec12 = tf.linalg.eigh(inv12_v11)
    else:
        v_eigv12, v_eig_vec12 = eigh_inv12_v11

    logdet_prod2 = tf.reduce_sum(tf.log(u_eigv12[:, None] * v_eigv12[None, :] + 1.))
    logdet_1 = logdet_12 + logdet_prod2

    u2_logdet = m * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(u2_tril)))) * 2.
    v2_logdet = n * tf.reduce_sum(tf.log(tf.abs(tf.matrix_diag_part(v2_tril)))) * 2.
    logdet_2 = u2_logdet + v2_logdet

    logdet_difference = logdet_2 + logdet_1
    const = tf.to_double(n * m)

    vec = mean11 + mean12 - mean2
    inverse_cov2_vec = tf.cholesky_solve(u2_tril, tf.transpose(tf.cholesky_solve(v2_tril, tf.transpose(vec))))
    mean_diff = tf.reduce_sum(vec * inverse_cov2_vec)

    tmp1 = tf.matrix_triangular_solve(
        u2_tril,
        tf.matrix_triangular_solve(tf.transpose(u12_tril), u_eig_vec12, lower=not lower_u12))
    diag1 = tf.reduce_sum(tf.square(tmp1), 0)

    tmp2 = tf.matrix_triangular_solve(
        v2_tril,
        tf.matrix_triangular_solve(tf.transpose(v12_tril), v_eig_vec12, lower=not lower_v12))
    diag2 = tf.reduce_sum(tf.square(tmp2), 0)

    trace = tf.reduce_sum((diag1[:, None] * diag2[None, :]) / (u_eigv12[:, None] * v_eigv12[None, :] + 1.))

    kl = 0.5 * (logdet_difference - const + trace + mean_diff)
    return kl