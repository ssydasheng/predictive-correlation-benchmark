import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from core.abstractVI import AbstractVI


class BBB(AbstractVI):
    def __init__(self, posterior, obs_var, input_dim, N, layer_sizes, eta=1., mini_particles=100, logstd_init=-5.):
        self.layer_sizes = layer_sizes
        self.logstd_init = logstd_init
        super(BBB, self).__init__(posterior, obs_var, input_dim, N, eta=eta, mini_particles=mini_particles)

    def build_params(self):
        self.q_params = {}
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1],
                                              self.layer_sizes[1:])):
            w_mean = tf.get_variable('w_mean_' + str(i), shape=[n_in, n_out],
                                     initializer=tf.random_normal_initializer())
            w_logstd = tf.get_variable('w_logstd_' + str(i), shape=[n_in, n_out],
                                       initializer=tf.constant_initializer(self.logstd_init))

            b_mean = tf.get_variable('b_mean_' + str(i), shape=[1, n_out],
                                     initializer=tf.zeros_initializer())
            b_logstd = tf.get_variable('b_logstd_' + str(i), shape=[1, n_out],
                                       initializer=tf.constant_initializer(self.logstd_init))

            self.q_params['w_' + str(i)] = (w_mean, w_logstd)
            self.q_params['b_' + str(i)] = (b_mean, b_logstd)

    def ws_bs(self, n_particles):
        ws, bs = [], []
        for i, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            w_mean, w_logstd = self.q_params['w_' + str(i)]
            w_std = tf.exp(w_logstd)
            w = w_mean + w_std * tf.random_normal([n_particles, n_in, n_out])

            b_mean, b_logstd = self.q_params['b_' + str(i)]
            b_std = tf.exp(b_logstd)
            b = b_mean + b_std * tf.random_normal([n_particles, 1, n_out])

            ws.append(w)
            bs.append(b)
        return ws, bs

    def build_function(self):
        self.build_params()
        ws, bs = self.ws_bs(self.n_particles)
        self.func_x = tf.squeeze(self.posterior(self.x, ws, bs, no_sample_dim=False), -1)
        self.func_x_pred = self.compute_func_x_pred_with_loop()
        self.func_x_pred = self.func_x_pred[:, :, 0]
        self.func_x_pred_mean, self.func_x_pred_var = tf.nn.moments(self.func_x_pred, 0)
        self.func_x_pred_std = tf.sqrt(self.func_x_pred_var)

        # ======== add covariance computation =========
        self.func_x_pred_cov = tfp.stats.covariance(self.func_x_pred, sample_axis=0, event_axis=-1)

    def build_log_likelihood(self):
        y_obs = tf.tile(tf.expand_dims(self.y, axis=0), [self.n_particles, 1])
        y_x_dist = tf.distributions.Normal(self.func_x, tf.to_float(self.obs_var)**0.5)
        self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
        self.log_likelihood = tf.reduce_mean(self.log_likelihood_sample)
        self.y_x_pred = y_x_dist.sample()

    def compute_func_x_pred_with_loop(self):
        def cond(i, func_x_pred):
            return tf.less(i, self.n_particles)

        def body(i, func_x_pred):
            i = tf.add(i, self.mini_particles)
            ws, bs = self.ws_bs(self.mini_particles)
            v = self.posterior(self.x_pred, ws, bs, no_sample_dim=False)
            func_x_pred = tf.concat([func_x_pred, v], 0)
            return i, func_x_pred

        func_x_pred = tf.zeros([1, tf.shape(self.x_pred)[0], 1])
        i = tf.constant(0)
        _, func_x_pred = tf.while_loop(cond, body, [i, func_x_pred], shape_invariants=[i.get_shape(), tf.TensorShape([None, None, 1])])
        return func_x_pred[1:]

    @property
    def params_posterior(self):
        return tf.trainable_variables('posterior')

    @property
    def params_prior(self):
        return tf.trainable_variables('prior')

    @property
    def params_likelihood(self):
        return tf.trainable_variables('likelihood')

    def get_prior_params(self, name, mean_q, var_q):
        mean_p = tf.zeros_like(mean_q)
        var_p = tf.ones_like(mean_q) * self.eta
        return mean_p, var_p

    def build_kl(self):
        kl = 0.
        for name, (mean_q, logstd_q) in self.q_params.items():
            mean_p, var_p = self.get_prior_params(name, mean_q, tf.exp(2.*logstd_q))
            kl = kl + tf.reduce_sum(tfp.distributions.kl_divergence(
                tfp.distributions.Normal(mean_q, tf.exp(logstd_q)),
                tfp.distributions.Normal(mean_p, tf.sqrt(var_p))
            ))
        self.kl = kl

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * tf.to_float(self.kl) / tf.to_float(self.N)

        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.infer_joint = self.optimizer.minimize(-self.elbo, global_step=self.global_step)

    def build_sep_optimizer(self):
        self.sep_optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        flag = len(self.params_likelihood)
        flag = flag and any([tf.gradients(-self.eval_lld, v)[0] is not None for v in self.params_likelihood])
        self.infer_sep_obs_var = self.sep_optimizer.minimize(-self.eval_lld, var_list=self.params_likelihood) \
            if flag else tf.no_op()
        self.infer_sep_others = self.sep_optimizer.minimize(
            -self.elbo, var_list=list(set(tf.trainable_variables())-set(self.params_likelihood)),
            global_step=self.global_step)