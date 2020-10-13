import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from core.abstract import Abstract


class Dropout(Abstract):
    def __init__(self, network, input_dim, N, test_particles=100, obs_var=0.025):
        super().__init__(input_dim)
        self.network = network
        self.N = N
        self.test_particles = test_particles
        self.obs_var = tf.constant(obs_var, dtype=tf.float32)
        self.build_()

    def build_(self):
        self.build_inputs()
        self.build_function()
        self.build_evaluation()
        self.build_optimizer()

    def build_function(self):
        self.func_x = tf.squeeze(self.network(self.x), -1)

        self.func_x_pred = self.compute_func_x_pred_with_loop()[..., 0]
        self.func_x_pred_mean, self.func_x_pred_var = tf.nn.moments(self.func_x_pred, 0)
        self.func_x_pred_std = tf.sqrt(self.func_x_pred_var)
        self.func_x_pred_cov = tfp.stats.covariance(self.func_x_pred, sample_axis=0, event_axis=-1)

    def compute_func_x_pred_with_loop(self):
        def cond(i, func_x_pred):
            return tf.less(i, self.test_particles)

        def body(i, func_x_pred):
            i = tf.add(i, 1.)
            v = self.network(self.x_pred, is_training=False)
            v = tf.expand_dims(v, 0)
            func_x_pred = tf.concat([func_x_pred, v], 0)
            return i, func_x_pred

        func_x_pred = tf.zeros([1, tf.shape(self.x_pred)[0], 1])
        i = tf.constant(0.)
        _, func_x_pred = tf.while_loop(cond, body, [i, func_x_pred], shape_invariants=[i.get_shape(), tf.TensorShape([None, None, 1])])
        return func_x_pred[1:]

    def build_optimizer(self):
        self.mse_loss = tf.reduce_mean(tf.square(self.func_x - self.y))
        self.l2_loss = tf.losses.get_regularization_loss()
        self.loss = self.mse_loss + self.l2_loss

        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.infer_joint = self.optimizer.minimize(self.loss, global_step=self.global_step)
