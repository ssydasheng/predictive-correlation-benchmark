# Copyright 2018 Shengyang Sun
# Copyright 2016 James Hensman, alexggmatthews
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

import tensorflow as tf
import numpy as np
import pandas as pd

from . import settings


__TRAINABLES = tf.GraphKeys.TRAINABLE_VARIABLES
__GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


def pretty_pandas_table(names, keys, values):
    df = pd.DataFrame(dict(zip(keys, values)))
    df.index = names
    df = df.reindex_axis(keys, axis=1)
    return df


def is_ndarray(value):
    return isinstance(value, np.ndarray)


def is_list(value):
    return isinstance(value, list)


def is_tensor(value):
    return isinstance(value, (tf.Tensor, tf.Variable))


def is_number(value):
    return (not isinstance(value, str)) and np.isscalar(value)


def is_valid_param_value(value):
    if isinstance(value, list):
        if not value:
            return False
        zero_val = value[0]
        arrays = (list, np.ndarray)
        scalars = (float, int)
        if isinstance(zero_val, scalars):
            types = scalars
        elif isinstance(zero_val, arrays):
            types = arrays
        else:
            return False
        return all(isinstance(val, types) for val in value[1:])
    return ((value is not None)
            and is_number(value)
            or is_ndarray(value)
            or is_tensor(value))

def normalize_num_type(num_type):
    """
    Work out what a sensible type for the array is. if the default type
    is float32, downcast 64bit float to float32. For ints, assume int32
    """
    if isinstance(num_type, tf.DType):
        num_type = num_type.as_numpy_dtype.type

    if num_type in [np.float32, np.float64]:  # pylint: disable=E1101
        num_type = settings.float_type
    elif num_type in [np.int16, np.int32, np.int64]:
        num_type = settings.int_type
    else:
        raise ValueError('Unknown dtype "{0}" passed to normalizer.'.format(num_type))

    return num_type


def vec_to_tri(vectors, N):
    """
    Takes a D x M tensor `vectors' and maps it to a D x matrix_size X matrix_sizetensor
    where the where the lower triangle of each matrix_size x matrix_size matrix is
    constructed by unpacking each M-vector.

    Native TensorFlow version of Custom Op by Mark van der Wilk.

    def int_shape(x):
        return list(map(int, x.get_shape()))

    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)


def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise

    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if var is None:
        return mean

    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND



def mvg_reparameterize(mean, z, u=None, v=None, u_sqrt=None, v_sqrt=None,
                       full_row_cov=False, full_col_cov=False):
    """
    Implements the 'reparameterization trick' for the MVG Gaussian of shape [S, N, D]

    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)

    If full_row_cov=True then u or u_sqrt must be of shape (S,N,N) and the full covariance is used for row.
    Otherwise u or u_sqrt must be (S,N) and the operation is elementwise

    If full_col_cov=True then v or v_sqrt must be of shape (S,D,D) and the full covariance is used for row.
    Otherwise v or v_sqrt must be (S,D) and the operation is elementwise

    :param mean: mean of shape (S,N,D)
    :param z: samples form unit Gaussian of shape S,N,D
    :param u: covariance of shape S,N,N or S,N
    :param u_sqrt: sqrt covariance of shape S,N,N or S,N
    :param v: covariance of shape S,N,N or S,D
    :param v_sqrt: sqrt covariance of shape S,D,D or S,D
    :param full_row_cov: bool to indicate whether u, u_sqrt is of shape S,N,N or S,N
    :param full_col_cov: bool to indicate whether v, v_sqrt is of shape S,D,D or S,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if u is None and u_sqrt is None:
        raise ValueError('u and u_sqrt cannot be None at the same time')

    if v is None and v_sqrt is None:
        raise ValueError('v and v_sqrt cannot be None at the same time')

    if u is not None and u_sqrt is not None:
        raise ValueError('u and u_sqrt cannot be not-None at the same time')

    if v is not None and v_sqrt is not None:
        raise ValueError('v and v_sqrt cannot be not-None at the same time')

    if full_row_cov is False and u is None:
        raise ValueError('when full_row_cov is False, u must be given')

    if full_col_cov is False and u is None:
        raise ValueError('when full_col_cov is False, v must be given')

    if full_row_cov is False and full_col_cov is False:
        var = tf.matmul(u[..., None], v[:, None, :])
        return mean + z * (var + settings.jitter) ** 0.5

    S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2]
    if full_row_cov and u_sqrt is None:
        u_sqrt = tf.cholesky(u + settings.jitter * tf.eye(N, dtype=settings.float_type))
    if full_col_cov and v_sqrt is None:
        v_sqrt = tf.cholesky(v + settings.jitter * tf.eye(D, dtype=settings.float_type))

    if full_row_cov is False and full_col_cov is True:
        return mean + tf.sqrt(u[..., None]) * tf.matmul(z, tf.transpose(v_sqrt))

    if full_row_cov is True and full_col_cov is False:
        return mean + tf.matmul(u_sqrt, z) * tf.sqrt(v[:, None])

    if full_row_cov is True and full_col_cov is True:
        return mean + tf.matmul(u_sqrt, z, tf.transpose(v_sqrt)) #TODO: transpose(u, v) must be checked

