import tensorflow as tf
import tensorflow_probability as tfp

from core.abstractVI import AbstractVI
from utils.kfac_ops import layer_collection as lc
from utils.kfac_ops.optimizer import NGOptimizer
from utils.kfac_ops.controller.weight_controller import WeightController
from utils.kfac_ops.utils import compute_eigmvg_stdnormal_kl_divergence
from utils.distributions import compute_mvg_kl_divergence


class NoisyKFAC(AbstractVI):
    def __init__(self, posterior, obs_var, input_dim, N, kl=1., eta=1., mini_particles=100):
        self.kl_coeff = kl
        self.mini_particles = mini_particles
        super(NoisyKFAC, self).__init__(posterior, obs_var, input_dim, N, eta=eta, mini_particles=mini_particles)

    def build_function(self):
        self.layer_collection = lc.LayerCollection(mode="kfac")
        self.sampler = WeightController(self.N, self.n_particles, kl=self.kl_coeff, eta=self.eta)

        self.func_x, self.l2_loss = self.posterior(self.x, self.n_particles, self.sampler,
                                                   self.layer_collection, self.obs_var, init=True)
        self.func_x = self.func_x[:, :, 0]
        # self.func_x_pred, _ = self.posterior(self.x_pred, self.n_particles,
        #                                   self.sampler, self.layer_collection, self.obs_var, init=False,
        #                                            norm_prior=self.norm_prior)

        self.func_x_pred = self.compute_func_x_pred_with_loop()

        self.func_x_pred = self.func_x_pred[:, :, 0]
        self.func_x_pred_mean, self.func_x_pred_var = tf.nn.moments(self.func_x_pred, 0)
        self.func_x_pred_std = tf.sqrt(self.func_x_pred_var)

        self.func_x_pred_cov = tfp.stats.covariance(self.func_x_pred, sample_axis=0, event_axis=-1)

    def compute_func_x_pred_with_loop(self):
        self.sampler.change_particles(self.mini_particles)
        def cond(i, func_x_pred):
            return tf.less(i, self.n_particles)

        def body(i, func_x_pred):
            i = tf.add(i, self.mini_particles)
            v, _ = self.posterior(self.x_pred, self.mini_particles, self.sampler, self.layer_collection,
                                  self.obs_var, init=False)
            func_x_pred = tf.concat([func_x_pred, v], 0)
            # func_x_pred = tf.Print(func_x_pred, [tf.shape(func_x_pred)], message='Func_x shape is:=')
            return i, func_x_pred

        func_x_pred = tf.zeros([1, tf.shape(self.x_pred)[0], 1])
        i = tf.constant(0)
        _, func_x_pred = tf.while_loop(cond, body, [i, func_x_pred],
                                       shape_invariants=[i.get_shape(), tf.TensorShape([None, None, 1])])
        self.sampler.change_particles(self.n_particles)
        return func_x_pred[1:]

    def build_log_likelihood(self):
        y_obs = tf.tile(tf.expand_dims(self.y, axis=0), [self.n_particles, 1])
        y_x_dist = tf.distributions.Normal(self.func_x, tf.to_float(self.obs_var)**0.5)
        self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
        self.log_likelihood = tf.reduce_mean(self.log_likelihood_sample)
        self.y_x_pred = y_x_dist.sample()

    @property
    def params_posterior(self):
        return tf.trainable_variables('posterior')

    @property
    def params_prior(self):
        return tf.trainable_variables('prior')

    @property
    def params_likelihood(self):
        return tf.trainable_variables('likelihood')

    def build_kl(self):
        pass

    def build_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.optim = NGOptimizer(var_list=self.params_posterior,
                                 learning_rate=1., # fix the learning rate
                                 cov_ema_decay=0.999, #TODO: enable tuning
                                 damping=1e-5,
                                 layer_collection=self.layer_collection,
                                 norm_constraint=1e-3,
                                 momentum=0.9,
                                 opt_type="kfac")

        coeff = self.kl_coeff / self.N
        n_params = tf.add_n([tf.to_float(tf.size(param)) for param in self.params_posterior])
        prior_loss = self.l2_loss / self.eta + tf.log(self.eta) * n_params * 0.5
        self.loss = -self.log_likelihood + coeff * prior_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            update_latent_op = self.optim.minimize(self.loss, global_step=self.global_step)
        with tf.control_dependencies([update_latent_op]):
            with tf.control_dependencies([self.optim.cov_update_op]):
                with tf.control_dependencies([self.optim.inv_update_op]):
                    self.infer_latent = self.sampler.update_weights(self.layer_collection.get_blocks())

        self.elbo = self.log_likelihood - self.kl_coeff / self.N * self.kl

        self.hyper_optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        flag = len(self.params_likelihood) or len(self.params_prior)
        params = self.params_likelihood + self.params_prior
        self.infer_hyper = self.hyper_optimizer.minimize(-self.elbo, var_list=params) if flag else tf.no_op()

        self.infer_joint = tf.group(self.infer_latent, self.infer_hyper)

    @property
    def kl(self):
        kl = 0.
        for layer in self.sampler._controller:
            n_in, n_out = layer._in_channels, layer._out_channels
            q_mu = layer._combine_weight_bias()
            kl += compute_mvg_kl_divergence(
                (q_mu, layer._u, layer._v),
                (tf.zeros_like(q_mu), tf.eye(n_in) * tf.sqrt(self.eta / n_in), tf.eye(n_out)),
                jitter=1e-8, sqrt_u1=True, sqrt_v1=True, sqrt_u2=True, sqrt_v2=True,
                lower_u1=False, lower_v1=False)
        return kl

    def build_sep_optimizer(self):
        return
