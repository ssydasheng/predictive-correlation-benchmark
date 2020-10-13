
from operator import mul, add
from functools import reduce
import core.gpflowSlim.kernels as gfsk
import core.gpflowSlim as gfs
from core.gpflowSlim.kernels import Kernel
import tensorflow as tf
import numpy as np
from core.gpflowSlim.params import Parameter
from core.gpflowSlim import transforms, settings


class NNGPReLUKernel(Kernel):
    def __init__(self,
                 input_dim,
                 vws,
                 vbs,
                 n_hiddens,
                 active_dims=None,
                 ARD=False,
                 name='NNGPKernel'):
        super(NNGPReLUKernel, self).__init__(input_dim, active_dims, name=name)
        self.ARD = ARD
        self.n_hiddens = n_hiddens
        with tf.variable_scope(name):
            var_ws = np.ones(n_hiddens+1, dtype=settings.float_type) * vws if ARD else vws
            var_bs = np.ones(n_hiddens+1, dtype=settings.float_type) * vbs if ARD else vbs

            self._var_ws = Parameter(var_ws, transform=transforms.positive, name='var_ws')
            self._var_bs = Parameter(var_bs, transform=transforms.positive, name='var_bs')

        self._parameters = self._parameters + [self._var_ws, self._var_bs]

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return self.compute_cov(X, X)
        else:
            XX = tf.concat([X, X2], 0)
            nX = tf.shape(X)[0]
            return self.compute_cov(XX, XX)[:nX, nX:]

    def Kdiag(self, X, presliced=False):
        return tf.linalg.diag_part(self.K(X, None, presliced))
    
    @property
    def var_bs(self):
        return self._var_bs.value
    
    @property
    def var_ws(self):
        return self._var_ws.value

    def compute_cov(self, X, X2):
        
        # M-by-N
        cov_x_x = tf.linalg.matmul(X, tf.transpose(X2))
 
        if self.ARD:
            K = self.var_bs[0] + self.var_ws[0] * (cov_x_x) / tf.cast(X.shape[1], cov_x_x.dtype)
        else:
            K = self.var_bs + self.var_ws * (cov_x_x) / tf.cast(X.shape[1], cov_x_x.dtype)
        for l in range(self.n_hiddens):
            if self.ARD:
                K = self.compute_gaussian_relu_expectation(K, self.var_bs[l+1], self.var_ws[l+1])
            else:
                K = self.compute_gaussian_relu_expectation(K, self.var_bs, self.var_ws)
        return K

    def compute_gaussian_relu_expectation(self, cov, var_b, var_w):
        # cov N-by-N
        
        EPS = 1e-8
        diag_cov = tf.linalg.diag_part(cov)
        diag_cov = tf.reshape(diag_cov, [-1, 1])  # 1-by-N
        det = tf.sqrt(-tf.pow(cov, 2) + tf.linalg.matmul(diag_cov, tf.transpose(diag_cov)) + EPS)   # N-by-N

    #     det = tf.print(det)
    #     cvdet = tf.print(cov / det)
        with tf.control_dependencies([tf.check_numerics(det, message='** First check: ')]):
            next_cov = 2 * det + cov * np.pi + 2 * cov * tf.math.atan(cov / det)  # N-by-N
        with tf.control_dependencies([tf.check_numerics(next_cov, message='** Second check: ')]):
            next_cov = next_cov / (4 * np.pi)
        return next_cov * var_w + var_b
    

class ExpDecay(Kernel):
    R"""
        The Exponential Decay kernel.
        .. math::
           k(x, x') = \frac{\beta^\alpha}{\left(x + x' + \beta \right)^\alpha}
    """
    def __init__(self,
                 input_dim,
                 alpha,
                 beta,
                 variance,
                 active_dims=None,
                 ARD=False,
                 name='kernel'):
        assert input_dim == 1, 'Currently only support 1 dim input'
        super().__init__(input_dim, active_dims, name=name)
        self.ARD =ARD
        with tf.variable_scope(name):
            variance = np.ones(self.input_dim, dtype=settings.float_type) * variance if ARD else variance
            self._variance = Parameter(variance, transform=transforms.positive, name='variance')

            alpha = np.ones(self.input_dim, dtype=settings.float_type) * alpha if ARD else alpha
            self._alpha = Parameter(alpha, transform=transforms.positive)

            beta = np.ones(self.input_dim, dtype=settings.float_type) * beta if ARD else beta
            self._beta = Parameter(beta, transform=transforms.positive)

        self._parameters = self._parameters + [self._variance, self._alpha, self._beta]

    @property
    def variance(self):
        return self._variance.value

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return tf.pow(self._beta, self._alpha) / tf.pow((X + tf.transpose(X)) + self._beta, self._alpha)
        else:
            return tf.pow(self._beta, self._alpha) / tf.pow((X + tf.transpose(X2)) + self._beta, self._alpha)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.pow(self._beta, self._alpha) / tf.pow(X + X + self._beta, self._alpha)


def SpectralMixture(params, name):
    """
    Build the SpectralMixture kernel.

    :params: list of dict. With each item corresponding to one mixture.
        The dict is formatted as {'w': float, 'rbf': dict, 'cos': dict}.
        That each sub-dict is used to the init corresponding kernel.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        sm = 0.
        for i in range(len(params)):
            w = gfs.Param(params[i]['w'], transform=gfs.transforms.positive, name='w' + str(i))
            sm = gfsk.RBF(**params[i]['rbf']) * gfsk.Cosine(**params[i]['cos']) * w.value + sm
        return sm


_KERNEL_DICT=dict(
    White=gfsk.White,
    Constant=gfsk.Constant,
    ExpQuad=gfsk.RBF,
    RBF=gfsk.RBF,
    Matern12=gfsk.Matern12,
    Matern32=gfsk.Matern32,
    Matern52=gfsk.Matern52,
    Cosine=gfsk.Cosine,
    ArcCosine=gfsk.ArcCosine,
    Linear=gfsk.Linear,
    Periodic=gfsk.Periodic,
    RatQuad=gfsk.RatQuad,
    
    SM=SpectralMixture,
    ExpDecay=ExpDecay
)

def KernelWrapper(hparams):
    """
    Wrapper for Kernels.
    :param hyparams: list of dict. Each item corresponds to one primitive kernel.
        The dict is formatted as {'name': XXX, 'params': XXX}.
        e.g.
            [{'name': 'Linear', params={'c': 0.1, 'input_dim': 100}},
             {'name': 'Periodic', params={'period': 2, 'input_dim': 100, 'ls': 2}}]
    """
    assert len(hparams) > 0, 'At least one kernel should be provided.'
    with tf.variable_scope('KernelWrapper'):
        return [_KERNEL_DICT[k['name']](**k['params']) for k in hparams]

