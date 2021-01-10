# Adapted from https://github.com/kekeblom/DeepCGP
import numpy as np
import tensorflow as tf


from .kernels import Kernel
from . import settings
from .params import Parameter


class AdditivePatchKernel(Kernel):
    """This conv kernel sums over each patch assuming the output is produced independently from each patch.
        K(x, x') = \sum_{i} w_i k(x[i], x'[i])
    """
    def __init__(self, base_kernel, view, patch_weights=None, name='AdditivePatch'):
        super().__init__(input_dim=np.prod(view.input_size), name=name)
        self.base_kernel = base_kernel
        self.view = view
        self.patch_length = view.patch_length
        self.patch_count = view.patch_count
        self.image_size = self.view.input_size
        if patch_weights is None or patch_weights.size != self.patch_count:
            patch_weights = np.ones(self.patch_count, dtype=settings.float_type)
        with tf.variable_scope(self._name):
            self._patch_weights = Parameter(patch_weights)
        self._parameters = self._parameters + self.base_kernel.parameters + [self._patch_weights]

    @property
    def patch_weights(self):
        return self._patch_weights.value

    def _reshape_X(self, ND_X):
        ND = tf.shape(ND_X)
        return tf.reshape(ND_X, [ND[0]] + list(self.view.input_size))

    def K(self, ND_X, X2=None):
        NHWC_X = self._reshape_X(ND_X)
        patch_length = self.patch_length
        PNL_patches = self.view.extract_patches_PNL(NHWC_X)

        if X2 is None:
            PNL_patches2 = patches
        else:
            PNL_patches2 = self.view.extract_patches_PNL(self._reshape_X(X2))

        def compute_K(tupled):
            NL_patches1, NL_patches2, weight = tupled
            return weight * self.base_kernel.K(NL_patches1, NL_patches2)

        PNN_K = tf.map_fn(compute_K, (PNL_patches, PNL_patches2, self.patch_weights), settings.float_type,
                parallel_iterations=self.patch_count)

        return tf.reduce_mean(PNN_K, [0])

    def Kdiag(self, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        PNL_patches = self.view.extract_patches_PNL(NHWC_X)
        def compute_Kdiag(tupled):
            NL_patches, weight = tupled
            return weight * self.base_kernel.Kdiag(NL_patches)
        PN_K = tf.map_fn(compute_Kdiag, (PNL_patches, self.patch_weights), settings.float_type,
                parallel_iterations=self.patch_count)
        return tf.reduce_mean(PN_K, [0])

    def Kzx(self, ML_Z, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        # Patches: N x patch_count x patch_length
        PNL_patches = self.view.extract_patches_PNL(NHWC_X)
        def compute_Kuf(tupled):
            NL_patches, weight = tupled
            return weight * self.base_kernel.K(ML_Z, NL_patches)

        KMN_Kuf = tf.map_fn(compute_Kuf, (PNL_patches, self.patch_weights), settings.float_type,
                parallel_iterations=self.patch_count)

        return tf.reduce_mean(KMN_Kuf, [0])

    def Kzz(self, Z):
        return self.base_kernel.K(Z)


class ConvKernel(AdditivePatchKernel):
    # Loosely based on https://github.com/markvdw/convgp/blob/master/convgp/convkernels.py
    def K(self, ND_X, X2=None):
        NHWC_X = self._reshape_X(ND_X)
        patch_length = self.patch_length
        # N * P x L
        patches = tf.reshape(self.view.extract_patches(NHWC_X), [-1, patch_length])

        if X2 is None:
            patches2 = patches
        else:
            patches2 = tf.reshape(self.view.extract_patches(X2), [-1, patch_length])

        # K: batch * patches x batch * patches
        K = self.base_kernel.K(patches, patches2)
        X_shape = tf.shape(NHWC_X)
        # Reshape to batch x patch_count x batch x patch_count
        K = tf.reshape(K, (X_shape[0], self.patch_count, X_shape[0], self.patch_count))

        w = self.patch_weights
        W = w[None, :] * w[:, None] # P x P
        W = W[None, :, None, :] # 1 x P x 1 x P
        K = K * W

        # Sum over the patch dimensions.
        return tf.reduce_sum(K, [1, 3]) / (self.patch_count ** 2)

    def Kdiag(self, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        # Compute auto correlation in the patch space.
        # patches: N x patch_count x patch_length
        patches = self.view.extract_patches(NHWC_X)
        w = self.patch_weights
        W = w[None, :] * w[:, None]
        def sumK(p):
            return tf.reduce_sum(self.base_kernel.K(p) * W)
        return tf.map_fn(sumK, patches, parallel_iterations=self.patch_count) / (self.patch_count ** 2)

    def Kzx(self, Z, ND_X):
        NHWC_X = self._reshape_X(ND_X)
        # Patches: N x patch_count x patch_length
        patches = self.view.extract_patches(NHWC_X)
        patches = tf.reshape(patches, (-1, self.patch_length))
        # Kzx shape: M x N * patch_count
        Kzx = self.base_kernel.K(Z, patches)
        M = tf.shape(Z)[0]
        N = tf.shape(NHWC_X)[0]
        # Reshape to M x N x patch_count then sum over patches.
        Kzx = tf.reshape(Kzx, (M, N, self.patch_count))

        w = self.patch_weights
        Kzx = Kzx * w

        Kzx = tf.reduce_sum(Kzx, [2])
        return Kzx / self.patch_count

    def Kzz(self, Z):
        return self.base_kernel.K(Z)
