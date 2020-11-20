# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews
# Copyright 2017 Artem Artemev @awav
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function, absolute_import
from functools import reduce
import warnings

import tensorflow as tf
#import tensorflow.contrib.eager as tfe
import numpy as np

from . import transforms
from . import settings

from .params import Parameter
from .quadrature import mvnquad


class Kernel(object):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        """
        input dim is an integer
        active dims is either an iterable of integers or None.

        Input dim is the number of input dimensions to the kernel. If the
        kernel is computed on a matrix X which has more columns than input_dim,
        then by default, only the first input_dim columns are used. If
        different columns are required, then they may be specified by
        active_dims.

        If active dims is None, it effectively defaults to range(input_dim),
        but we store it as a slice for efficiency.
        """
        self._name = name
        self.input_dim = int(input_dim)
        if active_dims is None:
            self.active_dims = slice(input_dim)
        elif isinstance(active_dims, slice):
            self.active_dims = active_dims
            if active_dims.start is not None and active_dims.stop is not None and active_dims.step is not None:
                assert len(range(*active_dims)) == input_dim  # pragma: no cover
        else:
            self.active_dims = np.array(active_dims, dtype=np.int32)
            assert len(active_dims) == input_dim

        self.num_gauss_hermite_points = 20
        self._parameters = []

    def compute_K(self, X, Z):
        return self.K(X, Z)

    def compute_K_symm(self, X):
        return self.K(X)

    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    def compute_eKdiag(self, X, Xcov=None):
        return self.eKdiag(X, Xcov)

    def compute_eKxz(self, Z, Xmu, Xcov):
        return self.eKxz(Z, Xmu, Xcov)

    def compute_exKxz_pairwise(self, Z, Xmu, Xcov):
        return self.exKxz_pairwise(Z, Xmu, Xcov)

    def compute_exKxz(self, Z, Xmu, Xcov):
        return self.exKxz(Z, Xmu, Xcov)

    def compute_eKzxKxz(self, Z, Xmu, Xcov):
        return self.eKzxKxz(Z, Xmu, Xcov)

    def eKdiag(self, Xmu, Xcov):
        """
        Computes <K_xx>_q(x).
        :param Xmu: Mean (NxD)
        :param Xcov: Covariance (NxDxD or NxD)
        :return: (N)
        """
        self._check_quadrature()
        Xmu, _ = self._slice(Xmu, None)
        Xcov = self._slice_cov(Xcov)
        return mvnquad(lambda x: self.Kdiag(x, presliced=True),
                       Xmu, Xcov,
                       self.num_gauss_hermite_points, self.input_dim)  # N

    def eKxz(self, Z, Xmu, Xcov):
        """
        Computes <K_xz>_q(x) using quadrature.
        :param Z: Fixed inputs (MxD).
        :param Xmu: X means (NxD).
        :param Xcov: X covariances (NxDxD or NxD).
        :return: (NxM)
        """
        self._check_quadrature()
        Xmu, Z = self._slice(Xmu, Z)
        Xcov = self._slice_cov(Xcov)
        M = tf.shape(Z)[0]
        return mvnquad(lambda x: self.K(x, Z, presliced=True), Xmu, Xcov, self.num_gauss_hermite_points,
                       self.input_dim, Dout=(M,))  # (H**DxNxD, H**D)

    def exKxz_pairwise(self, Z, Xmu, Xcov):
        """
        Computes <x_{t-1} K_{x_t z}>_q(x) for each pair of consecutive X's in
        Xmu & Xcov.
        :param Z: Fixed inputs (MxD).
        :param Xmu: X means (T+1xD).
        :param Xcov: 2xT+1xDxD. [0, t, :, :] contains covariances for x_t. [1, t, :, :] contains the cross covariances
        for t and t+1.
        :return: (TxMxD).
        """
        self._check_quadrature()
        # Slicing is NOT needed here. The desired behaviour is to *still* return an NxMxD matrix. As even when the
        # kernel does not depend on certain inputs, the output matrix will still contain the outer product between the
        # mean of x_{t-1} and K_{x_t Z}. The code here will do this correctly automatically, since the quadrature will
        # still be done over the distribution x_{t-1, t}, only now the kernel will not depend on certain inputs.
        # However, this does mean that at the time of running this function we need to know the input *size* of Xmu, not
        # just `input_dim`.
        M = tf.shape(Z)[0]
        D = self.input_size if hasattr(self, 'input_size') else self.input_dim  # Number of actual input dimensions

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], tf.constant(D, dtype=settings.tf_int),
                            message="Numerical quadrature needs to know correct shape of Xmu.")
        ]):
            Xmu = tf.identity(Xmu)

        # First, transform the compact representation of Xmu and Xcov into a
        # list of full distributions.
        fXmu = tf.concat((Xmu[:-1, :], Xmu[1:, :]), 1)  # Nx2D
        fXcovt = tf.concat((Xcov[0, :-1, :, :], Xcov[1, :-1, :, :]), 2)  # NxDx2D
        fXcovb = tf.concat((tf.transpose(Xcov[1, :-1, :, :], (0, 2, 1)), Xcov[0, 1:, :, :]), 2)
        fXcov = tf.concat((fXcovt, fXcovb), 1)
        return mvnquad(lambda x: tf.expand_dims(self.K(x[:, :D], Z), 2) *
                                 tf.expand_dims(x[:, D:], 1),
                       fXmu, fXcov, self.num_gauss_hermite_points,
                       2 * D, Dout=(M, D))

    def exKxz(self, Z, Xmu, Xcov):
        """
        Computes <x_t K_{x_t z}>_q(x) for the same x_t.
        :param Z: Fixed inputs (MxD).
        :param Xmu: X means (TxD).
        :param Xcov: TxDxD. Contains covariances for each x_t.
        :return: (TxMxD).
        """
        self._check_quadrature()
        # Slicing is NOT needed here. The desired behaviour is to *still* return an NxMxD matrix.
        # As even when the kernel does not depend on certain inputs, the output matrix will still
        # contain the outer product between the mean of x_t and K_{x_t Z}. The code here will
        # do this correctly automatically, since the quadrature will still be done over the
        # distribution x_t, only now the kernel will not depend on certain inputs.
        # However, this does mean that at the time of running this function we need to know the
        # input *size* of Xmu, not just `input_dim`.
        M = tf.shape(Z)[0]
        # Number of actual input dimensions
        D = self.input_size if hasattr(self, 'input_size') else self.input_dim

        msg = "Numerical quadrature needs to know correct shape of Xmu."
        assert_shape = tf.assert_equal(tf.shape(Xmu)[1], D, message=msg)
        with tf.control_dependencies([assert_shape]):
            Xmu = tf.identity(Xmu)

        def integrand(x):
            return tf.expand_dims(self.K(x, Z), 2) * tf.expand_dims(x, 1)

        num_points = self.num_gauss_hermite_points
        return mvnquad(integrand, Xmu, Xcov, num_points, D, Dout=(M, D))

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        Computes <K_zx Kxz>_q(x).
        :param Z: Fixed inputs MxD.
        :param Xmu: X means (NxD).
        :param Xcov: X covariances (NxDxD or NxD).
        :return: NxMxM
        """
        self._check_quadrature()
        Xmu, Z = self._slice(Xmu, Z)
        Xcov = self._slice_cov(Xcov)
        M = tf.shape(Z)[0]

        def KzxKxz(x):
            Kxz = self.K(x, Z, presliced=True)
            return tf.expand_dims(Kxz, 2) * tf.expand_dims(Kxz, 1)

        return mvnquad(KzxKxz,
                       Xmu, Xcov, self.num_gauss_hermite_points,
                       self.input_dim, Dout=(M, M))

    def _check_quadrature(self):
        if settings.numerics.ekern_quadrature == "warn":
            warnings.warn("Using numerical quadrature for kernel expectation of %s. Use gpflow.ekernels instead." %
                          str(type(self)))
        if settings.numerics.ekern_quadrature == "error" or self.num_gauss_hermite_points == 0:
            raise RuntimeError("Settings indicate that quadrature may not be used.")

    def _slice(self, X, X2):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims`.
        :param X: Input 1 (NxD).
        :param X2: Input 2 (MxD), may be None.
        :return: Sliced X, X2, (Nxself.input_dim).
        """
        # for tf.gather in self._slice
        if False:#tfe.in_eager_mode():
            active_dims = tf.constant(list(self.active_dims)).gpu()
        else:
            active_dims = self.active_dims
        #
        # if isinstance(X, tf.Tensor):
        #     if len(self.active_dims) == X.get_shape().as_list()[-1]:
        #         return X, X2
        # if isinstance(X, np.ndarray):
        #     if len(self.active_dims) == X.shape[-1]:
        #         return X, X2

        if isinstance(active_dims, slice):
            X = X[:, active_dims]
            if X2 is not None:
                X2 = X2[:, active_dims]
        else:
            X = tf.gather(X, active_dims, axis=1)
            if X2 is not None:
                X2 = tf.gather(X2, active_dims, axis=1)

        #TODO: wait for eager fix this bug.
        # input_dim_shape = tf.shape(X)[1]
        # input_dim = tf.convert_to_tensor(self.input_dim, dtype=settings.tf_int)
        # with tf.control_dependencies([tf.assert_equal(input_dim_shape, input_dim)]):
        #     X = tf.identity(X)

        return X, X2

    def _slice_cov(self, cov):
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.
        :param cov: Tensor of covariance matrices (NxDxD or NxD).
        :return: N x self.input_dim x self.input_dim.
        """
        cov = tf.cond(tf.equal(tf.rank(cov), 2), lambda: tf.matrix_diag(cov), lambda: cov)

        if isinstance(self.active_dims, slice):
            cov = cov[..., self.active_dims, self.active_dims]
        else:
            cov_shape = tf.shape(cov)
            covr = tf.reshape(cov, [-1, cov_shape[-1], cov_shape[-1]])
            gather1 = tf.gather(tf.transpose(covr, [2, 1, 0]), self.active_dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), self.active_dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat([cov_shape[:-2], [len(self.active_dims), len(self.active_dims)]], 0))
        return cov

    def __add__(self, other):
        return Sum([self, other])

    def __mul__(self, other):
        return Product([self, other])

    @property
    def parameters(self):
        return self._parameters

    def Kdim(self, dim, X, X2=None):
        """
        Compute the covariance along that dim.
        :param dim: Int
        :param X: [n, 1]
        :param X2: [n, 1] or None
        :return: [n, n]
        """
        X = tf.concat([
            tf.zeros([tf.shape(X)[0], dim], dtype=settings.tf_float),
            X,
            tf.zeros([tf.shape(X)[0], self.input_dim - dim - 1], dtype=settings.tf_float)
        ], axis=1)
        if X2 is not None:
            X2 = tf.concat([
                tf.zeros([tf.shape(X2)[0], dim], dtype=settings.tf_float),
                X2,
                tf.zeros([tf.shape(X2)[0], self.input_dim - dim - 1], dtype=settings.tf_float)
            ], axis=1)
        return self.K(X, X2)

class Static(Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)
        with tf.compat.v1.variable_scope(name):
            self._variance = Parameter(variance, transform=transforms.positive, name='variance')
        self._parameters = self._parameters + [self._variance]

    @property
    def variance(self):
        return self._variance.value

    def Kdiag(self, X):
        return tf.ones_like(X[:, 0], dtype=settings.float_type) * self.variance


class White(Static):
    """
    The White kernel
    """
    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            d = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))
            return tf.matrix_diag(d)
        else:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X2)[0]])
            return tf.zeros(shape, settings.float_type)


class Constant(Static):
    """
    The Constant (aka Bias) kernel
    """
    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X)[0]])
        else:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X2)[0]])
        return tf.fill(shape, tf.squeeze(self.variance))


class Bias(Constant):
    """
    Another name for the Constant kernel, included for convenience.
    """
    pass


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, min_ls=1e-6, name='kernel'):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        super().__init__(input_dim, active_dims, name=name)

        with tf.compat.v1.variable_scope(name):
            self._variance = Parameter(variance, transform=transforms.positive,
                                       name='variance', dtype=settings.float_type)
            if ARD:
                if lengthscales is None:
                    lengthscales = np.ones(input_dim, dtype=settings.float_type)
                else:
                    lengthscales = lengthscales * np.ones(input_dim, dtype=settings.float_type)
            else:
                lengthscales = 1.0 if lengthscales is None else lengthscales
            self.ARD = ARD
            self._ls = Parameter(lengthscales, transform=transforms.Log1pe(min_ls), name='ls')

        self._parameters = self._parameters + [self._variance, self._ls]

    @property
    def variance(self):
        return self._variance.value

    @property
    def lengthscales(self):
        return self._ls.value

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1))  + tf.reshape(Xs, (1, -1))
            return tf.clip_by_value(dist, 0., np.inf)

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return tf.clip_by_value(dist, 0., np.inf)


    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12) #TODO: GPflow uses 1e-36

    def Kdiag(self, X, presliced=False):
        return tf.ones_like(X[:, 0], dtype=settings.float_type) * self.variance


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)

    def dimwise(self, dim):
        return RBF(input_dim=1, variance=self.variance**(1./self.input_dim),
                   lengthscales=self.lengthscales[dim] if self.ARD else self.lengthscales,
                   name='RBF_dimwise_{}'.format(dim))


class RatQuad(Stationary):
    R"""
    The Rational Quadratic kernel.
    .. math::
       k(x, x') = \left(1 + \frac{(x - x')^2}{2\alpha\ell^2} \right)^{-\alpha}
    """
    def __init__(self, input_dim, alpha=1., variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, min_ls=1e-6, name='kernel'):
        with tf.compat.v1.variable_scope(name):
            super(RatQuad, self).__init__(
                input_dim=input_dim, variance=variance,
                lengthscales=lengthscales, active_dims=active_dims, ARD=ARD, min_ls=min_ls, name=name)
            self._alpha = Parameter(alpha, transform=transforms.positive, name='alpha')

        self._parameters = self._parameters + [self._alpha]

    @property
    def alpha(self):
        return self._alpha.value

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return self.variance * tf.pow((1.0 + 0.5 * self.square_dist(X, X2) *
                          (1.0 / self.alpha)), -1.0 * self.alpha)


class Linear(Kernel):
    """
    The linear kernel
    """

    def __init__(self, input_dim, variance=1.0, active_dims=None, ARD=False, name='kernel'):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        super().__init__(input_dim, active_dims, name=name)
        self.ARD = ARD
        with tf.compat.v1.variable_scope(name):
            variance = np.ones(self.input_dim, dtype=settings.float_type) * variance if ARD else variance
            self._variance = Parameter(variance, transform=transforms.positive, name='variance')

        self._parameters = self._parameters + [self._variance]

    @property
    def variance(self):
        return self._variance.value

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return tf.matmul(X * self.variance, X, transpose_b=True)
        else:
            return tf.matmul(X * self.variance, X2, transpose_b=True)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.reduce_sum(tf.square(X) * self.variance, 1)

    def dimwise(self, dim):
        return Linear(input_dim=1,
                      variance=self.variance[dim] if self.ARD else self.variance**(1./self.input_dim),
                      name='Linear_dimwise_{}'.format(dim))


class Polynomial(Linear):
    """
    The Polynomial kernel. Samples are polynomials of degree `d`.
    """

    def __init__(self, input_dim,
                 degree=3.0,
                 variance=1.0,
                 offset=1.0,
                 active_dims=None,
                 ARD=False,
                 name='kernel'):
        """
        :param input_dim: the dimension of the input to the kernel
        :param variance: the (initial) value for the variance parameter(s)
                         if ARD=True, there is one variance per input
        :param degree: the degree of the polynomial
        :param active_dims: a list of length input_dim which controls
          which columns of X are used.
        :param ARD: use variance as described
        """
        super().__init__(input_dim, variance, active_dims, ARD, name=name)
        self.degree = degree
        with tf.compat.v1.variable_scope(name):
            self._offset = Parameter(offset, transform=transforms.positive, name='offset')

        self._parameters = self._parameters + [self._variance, self._offset]

    @property
    def offset(self):
        return self._offset.value

    def K(self, X, X2=None, presliced=False):
        return (Linear.K(self, X, X2, presliced=presliced) + self.offset) ** self.degree

    def Kdiag(self, X, presliced=False):
        return (Linear.Kdiag(self, X, presliced=presliced) + self.offset) ** self.degree


class Exponential(Stationary):
    """
    The Exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-0.5 * r)


class Matern12(Stationary):
    """
    The Matern 1/2 kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-r)

    def dimwise(self, dim):
        return Matern12(input_dim=1, variance=self.variance**(1./self.input_dim),
                        lengthscales=self.lengthscales[dim] if self.ARD else self.lengthscales,
                        name='Matern12_dimwise_{}'.format(dim))


class Matern32(Stationary):
    """
    The Matern 3/2 kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * (1. + np.sqrt(3.) * r) * \
               tf.exp(-np.sqrt(3.) * r)

    def dimwise(self, dim):
        return Matern32(input_dim=1, variance=self.variance**(1./self.input_dim),
                        lengthscales=self.lengthscales[dim] if self.ARD else self.lengthscales,
                        name='Matern32_dimwise_{}'.format(dim))

class Matern52(Stationary):
    """
    The Matern 5/2 kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * (1.0 + np.sqrt(5.) * r + 5. / 3. * tf.square(r)) \
               * tf.exp(-np.sqrt(5.) * r)

    def dimwise(self, dim):
        return Matern52(input_dim=1, variance=self.variance**(1./self.input_dim),
                        lengthscales=self.lengthscales[dim] if self.ARD else self.lengthscales,
                        name='Matern52_dimwise_{}'.format(dim))

class Cosine(Stationary):
    """
    The Cosine kernel
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, min_ls=1e-6, name='kernel'):
        super(Cosine, self).__init__(input_dim, variance=variance, lengthscales=lengthscales,
                 active_dims=active_dims, ARD=ARD, min_ls=min_ls, name=name)
        with tf.compat.v1.variable_scope(name):
            self._weights = Parameter(np.random.normal(size=[input_dim, 1]), name='weights')
        self._parameters = self._parameters + [self._weights]

    @property
    def weights(self):
        return self._weights.value

    def K(self, X, X2=None, presliced=False):
        X = X / self.lengthscales
        if not presliced:
            X, X2 = self._slice(X, X2)
        prod = tf.squeeze(tf.matmul(X, self.weights), -1)
        if X2 is None:
            prod2 = prod
        else:
            X2 = X2 / self.lengthscales
            prod2 = tf.squeeze(tf.matmul(X2, self.weights), -1)
        prod = tf.expand_dims(prod, 1)
        prod2 = tf.expand_dims(prod2, 0)
        r = prod - prod2
        return self.variance * tf.cos(r)


class ArcCosine(Kernel):
    """
    The Arc-cosine family of kernels which mimics the computation in neural
    networks. The order parameter specifies the assumed activation function.
    The Multi Layer Perceptron (MLP) kernel is closely related to the ArcCosine
    kernel of order 0. The key reference is

    ::

        @incollection{NIPS2009_3628,
            title = {Kernel Methods for Deep Learning},
            author = {Youngmin Cho and Lawrence K. Saul},
            booktitle = {Advances in Neural Information Processing Systems 22},
            year = {2009},
            url = {http://papers.nips.cc/paper/3628-kernel-methods-for-deep-learning.pdf}
        }
    """

    implemented_orders = {0, 1, 2}
    def __init__(self, input_dim,
                 order=0,
                 variance=1.0, weight_variances=1., bias_variance=1.0,
                 active_dims=None, ARD=False, name='kernel'):
        """
        - input_dim is the dimension of the input to the kernel
        - order specifies the activation function of the neural network
          the function is a rectified monomial of the chosen order.
        - variance is the initial value for the variance parameter
        - weight_variances is the initial value for the weight_variances parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - bias_variance is the initial value for the bias_variance parameter
          defaults to 1.0.
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one weight_variance per dimension
          (ARD=True) or a single weight_variance (ARD=False).
        """
        super().__init__(input_dim, active_dims, name=name)

        if order not in self.implemented_orders:
            raise ValueError('Requested kernel order is not implemented.')
        self.order = order

        with tf.compat.v1.variable_scope(name):
            self._variance = Parameter(variance, transform=transforms.positive, name='variance')
            self._bias_variance = Parameter(bias_variance, transform=transforms.positive, name='bias_variance')
            if ARD:
                if weight_variances is None:
                    weight_variances = np.ones(input_dim, settings.float_type)
                else:
                    # accepts float or array:
                    weight_variances = weight_variances * np.ones(input_dim, settings.float_type)
            else:
                if weight_variances is None:
                    weight_variances = 1.0
            self.ARD = ARD
            self._weight_variances = Parameter(
                    weight_variances, transform=transforms.positive, name='weight_variances')

        self._parameters = self._parameters + [self._variance, self._bias_variance, self._weight_variances]

    @property
    def variance(self):
        return self._variance.value

    @property
    def bias_variance(self):
        return self._bias_variance.value

    @property
    def weight_variances(self):
        return self._weight_variances.value

    def _weighted_product(self, X, X2=None):
        if X2 is None:
            return tf.reduce_sum(self.weight_variances * tf.square(X), axis=1) + self.bias_variance
        return tf.matmul((self.weight_variances * X), X2, transpose_b=True) + self.bias_variance

    def _J(self, theta):
        """
        Implements the order dependent family of functions defined in equations
        4 to 7 in the reference paper.
        """
        if self.order == 0:
            return np.pi - theta
        elif self.order == 1:
            return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)
        elif self.order == 2:
            return 3. * tf.sin(theta) * tf.cos(theta) + \
                   (np.pi - theta) * (1. + 2. * tf.cos(theta) ** 2)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)

        X_denominator = tf.sqrt(self._weighted_product(X))
        if X2 is None:
            X2 = X
            X2_denominator = X_denominator
        else:
            X2_denominator = tf.sqrt(self._weighted_product(X2))

        numerator = self._weighted_product(X, X2)
        cos_theta = numerator / X_denominator[:, None] / X2_denominator[None, :]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)

        return self.variance * (1. / np.pi) * self._J(theta) * \
               X_denominator[:, None] ** self.order * \
               X2_denominator[None, :] ** self.order

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        X_product = self._weighted_product(X)
        theta = tf.constant(0., settings.float_type)
        return self.variance * (1. / np.pi) * self._J(theta) * X_product ** self.order


class Periodic(Kernel):
    """
    The periodic kernel. Defined in  Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    Derived using the mapping u=(cos(x), sin(x)) on the inputs.
    """

    def __init__(self, input_dim, period=1.0, variance=1.0,
                 lengthscales=1.0, active_dims=None, name='kernel'):
        # No ARD support for lengthscale or period yet
        super().__init__(input_dim, active_dims, name=name)

        with tf.compat.v1.variable_scope(name):
            self._variance = Parameter(variance, transform=transforms.positive, name='variance')
            self._ls = Parameter(lengthscales, transform=transforms.positive, name='ls')
            self._period = Parameter(period, transform=transforms.positive, name='period')

        self._parameters = self._parameters + [self._variance, self._ls, self._period]

    @property
    def variance(self):
        return self._variance.value

    @property
    def lengthscales(self):
        return self._ls.value

    @property
    def period(self):
        return self._period.value

    def Kdiag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        r = np.pi * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscales), 2)

        return self.variance * tf.exp(-0.5 * r)


class Coregion(Kernel):
    def __init__(self, input_dim, output_dim, rank, active_dims=None, name='kernel'):
        """
        A Coregionalization kernel. The inputs to this kernel are _integers_
        (we cast them from floats as needed) which usually specify the
        *outputs* of a Coregionalization model.

        The parameters of this kernel, W, kappa, specify a positive-definite
        matrix B.

          B = W W^T + diag(kappa) .

        The kernel function is then an indexing of this matrix, so

          K(x, y) = B[x, y] .

        We refer to the size of B as "num_outputs x num_outputs", since this is
        the number of outputs in a coregionalization model. We refer to the
        number of columns on W as 'rank': it is the number of degrees of
        correlation between the outputs.

        NB. There is a symmetry between the elements of W, which creates a
        local minimum at W=0. To avoid this, it's recommended to initialize the
        optimization (or MCMC chain) using a random W.
        """
        assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(input_dim, active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        with tf.compat.v1.variable_scope(name):
            self._W = Parameter(np.zeros((self.output_dim, self.rank), dtype=settings.float_type), name='W')
            self._kappa = Parameter(
                np.ones(self.output_dim, dtype=settings.float_type),
                transform=transforms.positive, name='kappa')

        self._parameters = self._parameters + [self._W, self._kappa]
    @property
    def W(self):
        return self._W.value

    @property
    def kappa(self):
        return self._kappa.value

    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        X = tf.cast(X[:, 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[:, 0], tf.int32)
        B = tf.matmul(self.W, self.W, transpose_b=True) + tf.matrix_diag(self.kappa)
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def Kdiag(self, X):
        X, _ = self._slice(X, None)
        X = tf.cast(X[:, 0], tf.int32)
        Bdiag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return tf.gather(Bdiag, X)


class CosineComplex(Stationary):
    """
    The Cosine kernel
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False, min_ls=1e-6, name='CosineComplex'):
        super(CosineComplex, self).__init__(input_dim, variance=variance, lengthscales=lengthscales,
                 active_dims=active_dims, ARD=ARD, min_ls=min_ls, name=name)
        with tf.compat.v1.variable_scope(name):
            self._weights = Parameter(np.random.normal(size=[input_dim, 1]), name='weights')
        self._parameters = self._parameters + [self._weights]

    @property
    def weights(self):
        return self._weights.value

    def K(self, X, X2=None, presliced=False):
        X = X / self.lengthscales
        if not presliced:
            X, X2 = self._slice(X, X2)
        prod = tf.squeeze(tf.matmul(X, self.weights), -1)
        if X2 is None:
            prod2 = prod
        else:
            X2 = X2 / self.lengthscales
            prod2 = tf.squeeze(tf.matmul(X2, self.weights), -1)
        prod = tf.expand_dims(prod, 1)
        prod2 = tf.expand_dims(prod2, 0)
        r = prod - prod2
        return tf.complex(self.variance * tf.cos(r), self.variance*tf.sin(r)) #self.variance*tf.sin(r)

    def Kdiag(self, X, presliced=False):
        real = tf.ones_like(X[:, 0], dtype=settings.float_type) * self.variance
        imag = tf.zeros_like(X[:, 0], dtype=settings.float_type) * self.variance
        return tf.complex(real, imag)



class RBFComplex(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        a = self.variance * tf.exp(-self.square_dist(X, X2) / 2)
        return tf.complex(a, tf.zeros_like(a, settings.tf_float))

    def dimwise(self, dim):
        return RBF(input_dim=1, variance=self.variance**(1./self.input_dim),
                   lengthscales=self.lengthscales[dim] if self.ARD else self.lengthscales,
                   name='RBF_dimwise_{}'.format(dim))

    def Kdiag(self, X, presliced=False):
        real = tf.ones_like(X[:, 0], dtype=settings.float_type) * self.variance
        imag = tf.zeros_like(X[:, 0], dtype=settings.float_type) * self.variance
        return tf.complex(real, imag)


class TPS(Stationary):

    def square_dist(self, X, X2):
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1))  + tf.reshape(Xs, (1, -1))
            return tf.clip_by_value(dist, 0., np.inf)

        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return tf.clip_by_value(dist, 0., np.inf)

    @property
    def R(self):
        return tf.constant(2., dtype=settings.float_type)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        D = tf.sqrt(self.square_dist(X, X2))
        kern = self.variance * (tf.pow(D, 3.) - 1.5*self.R*tf.square(D) + 0.5*tf.pow(self.R, 3.))
        return kern

    def Kdiag(self, X, presliced=False):
        return self.variance * 0.5*tf.pow(self.R, 3.) * tf.ones([tf.shape(X)[0]], dtype=settings.float_type)


def make_kernel_names(kern_list):
    """
    Take a list of kernels and return a list of strings, giving each kernel a
    unique name.

    Each name is made from the lower-case version of the kernel's class name.

    Duplicate kernels are given trailing numbers.
    """
    names = []
    counting_dict = {}
    for k in kern_list:
        inner_name = k.__class__.__name__.lower()

        # check for duplicates: start numbering if needed
        if inner_name in counting_dict:
            if counting_dict[inner_name] == 1:
                names[names.index(inner_name)] = inner_name + '_1'
            counting_dict[inner_name] += 1
            name = inner_name + '_' + str(counting_dict[inner_name])
        else:
            counting_dict[inner_name] = 1
            name = inner_name
        names.append(name)
    return names


class Combination(Kernel):
    """
    Combine a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the kernels to be combined are generated from their class
    names.
    """

    def __init__(self, kern_list, name='kernel'):
        #if not all(isinstance(k, Kernel) for k in kern_list):
        #    raise TypeError("can only combine Kernel instances")

        extra_dims = np.asarray([], dtype=int)
        active_dims = reduce(np.union1d, (np.r_[x.active_dims] for x in kern_list if isinstance(x, Kernel)), extra_dims)
        input_dim = active_dims.size

        super().__init__(input_dim=input_dim, name=name, active_dims=active_dims)

        # add kernels to a list, flattening out instances of this class therein
        self.kern_list = []
        self.const_list = []
        for k in kern_list:
            if isinstance(k, self.__class__):
                self.kern_list.extend(k.kern_list)
                self.const_list.extend(k.const_list)
            elif isinstance(k, (int, float, tf.Tensor)):
                self.const_list.append(k)
            else:
                self.kern_list.append(k)

        # generate a set of suitable names and add the kernels as attributes
        names = make_kernel_names(self.kern_list)
        [setattr(self, name, k) for name, k in zip(names, self.kern_list)]

        for kern in kern_list:
            if isinstance(kern, Kernel):
                self._parameters = self._parameters + kern.parameters

    @property
    def on_separate_dimensions(self):
        """
        Checks whether the kernels in the combination act on disjoint subsets
        of dimensions. Currently, it is hard to asses whether two slice objects
        will overlap, so this will always return False.
        :return: Boolean indicator.
        """
        if np.any([isinstance(k.active_dims, slice) for k in self.kern_list]):
            # Be conservative in the case of a slice object
            return False
        else:
            dimlist = [k.active_dims for k in self.kern_list]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1:]:
                    if np.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping


def _kernel_function(k, X, X2):
    if isinstance(k, (tf.Tensor, tf.Variable, int, float, np.ndarray)):
        return k
    return k.K(X, X2)

def _kernel_diag(k, X):
    if isinstance(k, (tf.Tensor,  tf.Variable, int, float, np.ndarray)):
        return k
    return k.Kdiag(X)


class Sum(Combination):
    def K(self, X, X2=None, presliced=False):
        return reduce(tf.add, [_kernel_function(k, X, X2) for k in self.kern_list] + self.const_list)

    def Kdiag(self, X, presliced=False):
        return reduce(tf.add, [_kernel_diag(k, X) for k in self.kern_list] + self.const_list)


class Product(Combination):
    def K(self, X, X2=None, presliced=False):
        return reduce(tf.multiply, [_kernel_function(k, X, X2) for k in self.kern_list] + self.const_list)

    def Kdiag(self, X, presliced=False):
        return reduce(tf.multiply, [_kernel_diag(k, X) for k in self.kern_list] + self.const_list)


def make_deprecated_class(oldname, NewClass):
    """
    Returns a class that raises NotImplementedError on instantiation.
    e.g.:
    >>> Kern = make_deprecated_class("Kern", Kernel)
    """
    msg = ("{module}.{} has been renamed to {module}.{}"
           .format(oldname, NewClass.__name__, module=NewClass.__module__))

    class OldClass(NewClass):
        def __new__(cls, *args, **kwargs):
            raise NotImplementedError(msg)
    OldClass.__doc__ = msg
    OldClass.__qualname__ = OldClass.__name__ = oldname
    return OldClass

Kern = make_deprecated_class("Kern", Kernel)
Add = make_deprecated_class("Add", Sum)
Prod = make_deprecated_class("Prod", Product)
PeriodicKernel = make_deprecated_class("PeriodicKernel", Periodic)
