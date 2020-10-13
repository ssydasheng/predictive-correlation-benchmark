import tensorflow as tf
from utils.common_utils import reuse_variables
# import zhusuan as zs


def get_posterior(name):
    if name == 'nn' or name == 'nn_relu':
        return MLP(tf.nn.relu)
    if name == 'nn_tanh':
        return MLP(tf.nn.tanh)
    if name == 'nn_input' or name == 'nn_input_relu':
        return mlp_input_outer(tf.nn.relu)
    raise NameError('Not a supported name for posterior')


def MLP(activation, scope='posterior'):
    def nn_inner(layer_sizes, use_dropout=True, dropout_rate=0.1, regularization=0., dropout_share_mask=False):
        @reuse_variables(scope)
        def mlp(x, is_training=True):
            h = x
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                if use_dropout:
                    if not dropout_share_mask or is_training:
                        h = tf.layers.dropout(h, dropout_rate, training=True)
                    else:
                        h = tf.layers.dropout(h, dropout_rate, training=True,
                                              noise_shape=tf.constant([1, n_in], dtype=tf.int32))
                h = tf.layers.dense(
                    h, n_out, activation=None,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(regularization))
                if i < len(layer_sizes) - 2:
                    h = activation(h)
            return h
        return mlp
    return nn_inner


def _append_homog(tensor):
    rank = len(tensor.shape.as_list())
    shape = tf.concat([tf.shape(tensor)[:-1], [1]], axis=0)
    ones = tf.ones(shape, dtype=tensor.dtype)
    return tf.concat([tensor, ones], axis=rank - 1)


def _dense(inputs, weights, activation, is_training=True, batch_norm=False, particles=1):
    inputs = _append_homog(inputs)
    n_in = inputs.shape.as_list()[-1]
    inputs = tf.reshape(inputs, [particles, -1, n_in])
    preactivations = tf.matmul(inputs, weights)
    preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])

    if batch_norm:
        bn = tf.layers.batch_normalization(preactivations, training=is_training)
        activations = activation(bn)
    else:
        activations = activation(preactivations)

    return preactivations, activations


def KFAC_MLP(activation, scope='posterior'):
    def nn_inner(layer_sizes, norm_fw=True):
        @reuse_variables(scope)
        def mlp(x, n_particles, controller, layer_collection, obs_var, init=True, regularization=0.0):
            h = tf.tile(x[None, ...], [n_particles, 1, 1])
            h = tf.reshape(h, [-1, layer_sizes[0]])
            preact, l2_loss = None, 0.
            for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1],
                                                  layer_sizes[1:])):
                if norm_fw: eta = 1. / (n_in + 1.)
                else: eta = 1.
                if init:
                    container, idx = controller.register_fc_layer(n_in, n_out, eta)
                    assert idx == i, 'idx=%d, i=%d' % (idx, i)
                sampled_weight = controller.get_weight(i)
                preact, act = _dense(h, sampled_weight, activation, particles=n_particles)
                l2_loss += 0.5 * tf.reduce_sum(tf.reduce_mean(sampled_weight ** 2, 0)) / eta
                if init:
                    layer_collection.register_fully_connected(controller.get_params(i), h, preact)
                h = act
            if init:
                layer_collection.register_normal_predictive_distribution(mean=preact, var=obs_var) #TODO: whether delte last dim
            preact = tf.reshape(preact, [n_particles, -1, layer_sizes[-1]])
            return preact, l2_loss
        return mlp
    return nn_inner


def mlp_input_outer(activation, scope='posterior'):
    def mlp_inner(layer_sizes, norm_fw=True):
        @reuse_variables(scope)
        def mlp(x, weights, biases, no_sample_dim=True):
            assert len(weights) == len(layer_sizes)-1
            assert len(biases) == len(layer_sizes)-1
            # x: [batch_size, input_dim]
            if no_sample_dim: h = x
            else: h = tf.tile(x[None, ...], [tf.shape(weights[0])[0], 1, 1])
            for i, (w, b) in enumerate(zip(weights, biases)):
                if norm_fw: coeff = tf.sqrt(tf.to_float(tf.shape(weights[i])[-2])+1.)
                else: coeff = 1.
                h = (tf.matmul(h, w) + b) / coeff
                if i < len(layer_sizes) - 2:
                    h = activation(h)
            # h: [n_particles, N, output_dim]
            return h
        return mlp
    return mlp_inner
