import core.gpflowSlim.kernels as gfsk
import core.gpflowSlim as gfs
import tensorflow as tf
from core.gpflowSlim.kernels import Kernel
import numpy as np
from core.gpflowSlim.params import Parameter
from core.gpflowSlim import transforms, settings


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
                 variance=1,
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
            self._alpha = Parameter(alpha, transform=transforms.positive, name='alpha')

            beta = np.ones(self.input_dim, dtype=settings.float_type) * beta if ARD else beta
            self._beta = Parameter(beta, transform=transforms.positive, name='beta')

        self._parameters = self._parameters + [self._variance, self._alpha, self._beta]

    @property
    def variance(self):
        return self._variance.value

    @property
    def alpha(self):
        return self._alpha.value

    @property
    def beta(self):
        return self._beta.value

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return tf.pow(self.beta, self.alpha) / tf.pow((X + tf.transpose(X)) + self.beta, self.alpha)
        else:
            return tf.pow(self.beta, self.alpha) / tf.pow((X + tf.transpose(X2)) + self.beta, self.alpha)

    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)
        return tf.reduce_sum(tf.pow(self.beta / (X + X + self.beta), self.alpha), 1)


_KERNEL_DICT = dict(
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
    ExpDecay=ExpDecay,
    SM=SpectralMixture,
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