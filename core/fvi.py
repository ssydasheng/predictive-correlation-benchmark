import tensorflow as tf
import tensorflow_probability as tfp

from utils.utils import variational_expectations
from core.abstractVI import AbstractVI


class AbstractFVI(AbstractVI):
    def __init__(self, posterior, rand_generator, obs_var, input_dim, n_rand, N):
        self._rand_generator = rand_generator
        self.n_rand = n_rand
        super().__init__(posterior, obs_var, input_dim, N, eta=None)

    def build_(self):
        self.build_inputs()
        self.build_coeff()
        self.build_rand()
        self.build_function()
        self.build_log_likelihood()
        self.build_evaluation()
        self.build_kl()
        self.build_optimizer()
        self.build_sep_optimizer()

    def build_rand(self):
        self.rand = self._rand_generator(self)
        self.x_rand = tf.concat([self.x, self.rand], axis=0)


class AbstractImplicitFVI(AbstractFVI):
    """
    Base Class for Functional Variational Inference, with posterior generating samples: Implicit posterior.

    :param posterior: the posterior network to be optimized.
    :param rand_generator: Generates measurement points.
    :param obs_var: Float. Observation variance.
    :param input_dim. Int.
    :param n_rand. Int. Number of random measurement points.
    """
    def build_function(self):
        # [n_particles, batch_size + n_rand]
        self.func_x_rand = self.posterior(self.x_rand, self.n_particles)
        self.func_x = self.func_x_rand[:, :tf.shape(self.x)[0]]

        self.func_x_pred = self.posterior(self.x_pred, self.n_particles)
        self.func_x_pred_mean, self.func_x_pred_var = tf.nn.moments(self.func_x_pred, 0)
        self.func_x_pred_std = tf.sqrt(self.func_x_pred_var)
        self.func_x_pred_cov = tfp.stats.covariance(self.func_x_pred, sample_axis=0, event_axis=-1)

    def build_log_likelihood(self):
        y_obs = tf.tile(tf.expand_dims(self.y, axis=0), [self.n_particles, 1])
        y_x_dist = tf.distributions.Normal(self.func_x, tf.to_float(self.obs_var)**0.5)
        self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
        self.log_likelihood = tf.reduce_mean(self.log_likelihood_sample)
        self.y_x_pred = y_x_dist.sample()

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * self.kl_surrogate / self.batch_size

        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        self.infer_latent = self.optimizer.minimize(-self.elbo, var_list=self.params_posterior,
                                                    global_step=self.global_step)

        flag = len(self.params_prior) and any([a is not None for a in tf.gradients(self.elbo, self.params_prior)])
        self.infer_prior = self.optimizer.minimize(-self.elbo, var_list=self.params_prior) if flag else tf.no_op()

        self.infer_likelihood = self.optimizer.minimize(-self.elbo, var_list=self.params_likelihood)\
            if len(self.params_likelihood) else tf.no_op()

        self.infer_joint = self.optimizer.minimize(-self.elbo, global_step=self.global_step)

    def build_sep_optimizer(self):
        raise NotImplementedError


class AbstractExplicitFVI(AbstractFVI):
    """
    Base Class for Function Variational Inference, with posterior generating distributions: Explicit posterior.
    """

    def build_log_likelihood(self):
        log_likelihood = variational_expectations(self.func_x_mean, self.func_x_std**2., self.y, self.obs_var)
        self.log_likelihood = tf.reduce_mean(log_likelihood)

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * tf.to_float(self.kl) / tf.to_float(self.N)

        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        self.infer_latent = self.optimizer.minimize(-self.elbo, var_list=self.params_posterior,
                                                    global_step=self.global_step)

        self.infer_joint = self.optimizer.minimize(-self.elbo, global_step=self.global_step)

    def build_sep_optimizer(self):
        self.sep_optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        flag = len(self.params_likelihood)
        flag = flag and any([v is not None for v in tf.gradients(self.eval_lld, self.params_likelihood)])
        self.infer_sep_obs_var = self.sep_optimizer.minimize(-self.eval_lld, var_list=self.params_likelihood) \
            if flag else tf.no_op()
        self.infer_sep_others = self.sep_optimizer.minimize(
            -self.elbo, var_list=list(set(tf.trainable_variables())-set(self.params_likelihood)),
            global_step=self.global_step)
