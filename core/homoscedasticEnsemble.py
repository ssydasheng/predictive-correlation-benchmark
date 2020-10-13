import tensorflow as tf
import tensorflow_probability as tfp

from core.abstract import Abstract

class HomoEnsembleNN(Abstract):
    def __init__(self, networks, input_dim, N, obs_var):
        super().__init__(input_dim)
        self.networks = networks
        self.N = N
        self.n_net = len(self.networks)
        self.obs_var = obs_var

        self.build_()

    def build_(self):
        self.build_inputs()
        self.build_function()
        self.build_evaluation()
        self.build_optimizer()

    def build_function(self):
        EPS = 1e-5
        single_batch_size = tf.to_int32(tf.shape(self.x)[0] / self.n_net)
        last = {self.n_net - 1: tf.shape(self.x)[0] - single_batch_size*self.n_net}
        # [batch_size, 2]
        self.func_x = tf.concat([net(self.x[single_batch_size * id: single_batch_size * (id+1) + last.get(id, 0)])
                      for id, net in enumerate(self.networks)], 0)
        self.func_x = tf.squeeze(self.func_x, [-1])

        # [n_net, batch_size, 2]
        self.func_x_pred = tf.stack([net(self.x_pred) for id, net in enumerate(self.networks)])
        self.func_x_pred = tf.squeeze(self.func_x_pred, [-1])
        # self._func_x_pred_means, self._func_x_pred_vars = outputs[..., 0], tf.exp(2.*outputs[..., 1])
        self.func_x_pred_mean, self.func_x_pred_var = tf.nn.moments(self.func_x_pred, 0)

        self.func_x_pred_std = tf.sqrt(self.func_x_pred_var) + EPS
        self.func_x_pred_cov = tfp.stats.covariance(self.func_x_pred, sample_axis=0, event_axis=-1)


    def build_optimizer(self):
        self.mse_loss = tf.reduce_mean(tf.square(self.func_x - self.y))
        self.l2_loss = tf.losses.get_regularization_loss()
        self.loss = self.mse_loss + self.l2_loss
        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.infer_joint = self.optimizer.minimize(self.loss, global_step=self.global_step)

