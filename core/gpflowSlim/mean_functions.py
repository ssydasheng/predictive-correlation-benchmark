# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, alexggmatthews, PabloLeon, Valentine Svensson
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


import tensorflow as tf
import numpy as np

from . import settings
from .params import Parameter


class MeanFunction(object):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """
    def __init__(self, name='MeanFunction'):
        self._parameters = []
        self._name = name

    def __call__(self, X):
        raise NotImplementedError("Implement the __call__ method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)

    @property
    def parameters(self):
        return self._parameters

    @property
    def name(self):
        return self._name


class Zero(MeanFunction):
    def __call__(self, X):
        return tf.zeros(tf.stack([tf.shape(X)[0], 1]), dtype=settings.float_type)


class Customized(MeanFunction):
    def __init__(self):
        W1 = np.ones((1, 20), dtype=settings.float_type)
        b1 = np.zeros(20, dtype=settings.float_type)
        W2 = np.ones((20, 20), dtype=settings.float_type)
        b2 = np.zeros(20, dtype=settings.float_type)
        W3 = np.ones((20, 1), dtype=settings.float_type)
        b3 = np.zeros(1, dtype=settings.float_type)
        MeanFunction.__init__(self)

        with tf.variable_scope(self.name):
            self._W1 = Parameter(np.atleast_2d(W1), name='W1')
            self._b1 = Parameter(b1, name='b1')
            self._W2 = Parameter(np.atleast_2d(W2), name='W2')
            self._b2 = Parameter(b2, name='b2')
            self._W3 = Parameter(np.atleast_2d(W3), name='W3')
            self._b3 = Parameter(b3, name='b3')
        self._parameters = self._parameters + [self._W1, self._b1,
                                               self._W2, self._b2,
                                               self._W3, self._b3]

    def __call__(self, X):
        h = tf.matmul(X, self._W1.value) + self._b1.value
        h = tf.nn.relu(h)
        h = tf.matmul(h, self._W2.value) + self._b2.value
        h = tf.nn.relu(h)
        h = tf.matmul(h, self._W3.value) + self._b3.value
        return h

class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=None, b=None, trainable=True, name='LinearMean'):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = np.ones((1, 1)) if A is None else A
        b = np.zeros(1) if b is None else b
        MeanFunction.__init__(self, name)

        with tf.variable_scope(self.name):
            self._A = Parameter(np.atleast_2d(A), name='A', trainable=trainable)
            self._b = Parameter(b, name='b', trainable=trainable)
        self._parameters = self._parameters + [self._A, self._b]

    @property
    def A(self):
        return self._A.value

    @property
    def b(self):
        return self._b.value

    def __call__(self, X):
        return tf.matmul(X, self.A) + self.b


class Identity(Linear):
    """
    y_i = x_i
    """
    def __init__(self, input_dim=None, name='Identity'):
        Linear.__init__(self, name=name)
        self.input_dim = input_dim

    def __call__(self, X):
        return X

    @property
    def A(self):
        if self.input_dim is None:
            raise ValueError("An input_dim needs to be specified when using the "
                             "`Identity` mean function in combination with expectations.")

        return tf.eye(self.input_dim, dtype=settings.float_type)

    @property
    def b(self):
        if self.input_dim is None:
            raise ValueError("An input_dim needs to be specified when using the "
                             "`Identity` mean function in combination with expectations.")

        return tf.zeros(self.input_dim, dtype=settings.float_type)


class Constant(MeanFunction):
    """
    y_i = c,,
    """
    def __init__(self, c=None):
        MeanFunction.__init__(self)
        c = np.zeros(1) if c is None else c
        with tf.variable_scope(self.name):
            self._c = Parameter(c, name='c')
        self._parameters = self._parameters + [self._c]

    @property
    def c(self):
        return self._c.value

    def __call__(self, X):
        shape = tf.stack([tf.shape(X)[0], 1])
        return tf.tile(tf.reshape(self.c, (1, -1)), shape)


class IdentityConv2dMean(MeanFunction):
    def __init__(self, filter_size, feature_maps_in, feature_maps_out=1, stride=1,
                 trainable=False, name='IdentityConv2dMean'):
        super().__init__(name=name)
        self.trainable = trainable
        self.filter_size = filter_size
        self.feature_maps_in = feature_maps_in
        self.feature_maps_out = feature_maps_out
        self.stride = stride
        with tf.variable_scope(self.name):
            self._conv_filter = Parameter(self._init_filter(), trainable=trainable)
        self._parameters = [self._conv_filter]

    @property
    def conv_filter(self):
        return self._conv_filter.value

    def __call__(self, NHWC_X):
        return tf.nn.conv2d(
            NHWC_X, self.conv_filter,
            strides=[1, self.stride, self.stride, 1],
            padding="VALID",
            data_format="NHWC")

    def _init_filter(self):
        identity_filter = np.zeros((self.filter_size, self.filter_size,
            self.feature_maps_in, self.feature_maps_out), dtype=settings.float_type)
        identity_filter[self.filter_size // 2, self.filter_size // 2, :, :] = 1.0
        return identity_filter


class Conv2dMean(IdentityConv2dMean):
    """The first filter map copies the center most pixel in the input and the rest are zero mean."""
    def _init_filter(self):
        # Only supports square filters with odd size for now. The first feature maps copies the input,
        # the rest is zero mean.
        identity_filter = np.zeros((self.filter_size, self.filter_size,
            self.feature_maps_in, self.feature_maps_out), dtype=settings.float_type)
        identity_filter[self.filter_size // 2, self.filter_size // 2, 0, 0] = 1.0
        return identity_filter

    def __call__(self, NHWC_X):
        value = super().__call__(NHWC_X)
        N = tf.shape(NHWC_X)[0]
        return tf.reshape(value, [N, -1])


class PatchwiseConv2d(Conv2dMean):
    def __init__(self, filter_size, feature_maps_in, out_height, out_width, name='PatchwiseConv2d'):
        super().__init__(filter_size, feature_maps_in, name=name)
        self.out_height = out_height
        self.out_width = out_width

    def __call__(self, PNL_patches):
        kernel = tf.reshape(self.conv_filter, [self.filter_size**2 * self.feature_maps_in,
            self.feature_maps_in])
        def foreach_patch(NL_patches):
            return tf.matmul(NL_patches, kernel)
        PN1_out = tf.map_fn(foreach_patch, PNL_patches)
        PNL = tf.shape(PNL_patches)
        return tf.reshape(tf.transpose(PN1_out, [2, 1, 0]), [PNL[1], PNL[0]])


class SwitchedMeanFunction(MeanFunction):
    pass
    #TODO: Implement this
    # """
    # This class enables to use different (independent) mean_functions respective
    # to the data 'label'.
    # We assume the 'label' is stored in the extra column of X.
    # """
    # def __init__(self, meanfunction_list):
    #     MeanFunction.__init__(self)
    #     for m in meanfunction_list:
    #         assert isinstance(m, MeanFunction)
    #     self.meanfunction_list = ParamList(meanfunction_list)
    #     self.num_meanfunctions = len(self.meanfunction_list)
    #
    # @params_as_tensors
    # def __call__(self, X):
    #     ind = tf.gather(tf.transpose(X), tf.shape(X)[1]-1)  # ind = X[:,-1]
    #     ind = tf.cast(ind, tf.int32)
    #     X = tf.transpose(tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1]-1)))  # X = X[:,:-1]
    #
    #     # split up X into chunks corresponding to the relevant likelihoods
    #     x_list = tf.dynamic_partition(X, ind, self.num_meanfunctions)
    #     # apply the likelihood-function to each section of the data
    #     results = [m(x) for x, m in zip(x_list, self.meanfunction_list)]
    #     # stitch the results back together
    #     partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_meanfunctions)
    #     return tf.dynamic_stitch(partitions, results)


class Additive(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X):
        return tf.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X):
        return tf.multiply(self.prod_1(X), self.prod_2(X))

