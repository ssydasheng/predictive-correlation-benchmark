
import tensorflow as tf
import tensorflow_probability as tfp
import core.gpflowSlim as gfs

from core.fvi import AbstractExplicitFVI


class GaussianFVI(AbstractExplicitFVI):
    def __init__(self, prior_kernel, posterior, rand_generator, obs_var,
                 input_dim, n_rand, N):
        self.prior_kernel = prior_kernel
        super(GaussianFVI, self).__init__(
            posterior, rand_generator, obs_var, input_dim, n_rand, N)

    def build_function(self):
        # [batch_size + n_rand]
        self.func_x_rand_mean, self.func_x_rand_cov = self.posterior(self.x_rand)
        self.func_x_pred_mean, self.func_x_pred_var = self.posterior(self.x_pred, full_cov=False)
        _, self.func_x_pred_cov = self.posterior(self.x_pred)

        self.func_x_mean, self.func_x_var = self.posterior(self.x, full_cov=False)
        self.func_x_mean = tf.to_float(self.func_x_mean)
        self.func_x_std = tf.to_float(tf.sqrt(self.func_x_var))

        self.func_x_pred_mean = tf.to_float(self.func_x_pred_mean)
        self.func_x_pred_std = tf.to_float(tf.sqrt(self.func_x_pred_var))
        self.func_x_pred = tf.transpose(self.func_x_pred_mean[..., None] + tf.matmul(
            tf.to_float(tf.cholesky(self.func_x_pred_cov)),
            tf.random_normal(shape=[tf.shape(self.x_pred)[0], self.n_particles])))

    def build_kl(self):
        jitter = gfs.settings.jitter * tf.eye(tf.shape(self.x_rand)[0], dtype=tf.float64)
        q = tfp.distributions.MultivariateNormalFullCovariance(
            tf.to_double(self.func_x_rand_mean),
            tf.to_double(self.func_x_rand_cov)
        )
        p = tfp.distributions.MultivariateNormalFullCovariance(
            tf.zeros_like(self.func_x_rand_mean, dtype=tf.float64),
            self.prior_kernel.K(tf.to_double(self.x_rand)) + jitter
        )
        self.kl = tf.to_float(tfp.distributions.kl_divergence(q, p))


class MiniBatchGaussianFVI(GaussianFVI):
    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * tf.to_float(self.kl) / tf.to_float(self.m)

        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)

        self.infer_latent = self.optimizer.minimize(-self.elbo, var_list=self.params_posterior,
                                                    global_step=self.global_step)

        self.infer_joint = self.optimizer.minimize(-self.elbo, global_step=self.global_step)
