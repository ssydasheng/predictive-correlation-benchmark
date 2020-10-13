import tensorflow as tf

from core.abstract import Abstract

class AbstractVI(Abstract):
    def __init__(self, posterior, obs_var, input_dim, N, eta=1., mini_particles=100):
        super().__init__(input_dim)
        self.mini_particles = mini_particles
        self.posterior = posterior
        self.obs_var = obs_var
        self.eta = eta
        self.N = N

        self.build_()

    def build_(self):
        self.build_inputs()
        self.build_coeff()
        self.build_function()
        self.build_log_likelihood()
        self.build_evaluation()
        self.build_kl()
        self.build_optimizer()
        self.build_sep_optimizer()

    def build_coeff(self):
        self.coeff_ll = tf.placeholder(tf.float32, shape=[], name='coeff_ll')
        self.coeff_kl = tf.placeholder(tf.float32, shape=[], name='coeff_kl')

    @property
    def batch_size(self):
        return tf.to_float(tf.shape(self.x)[0])

    def build_log_likelihood(self):
        raise NotImplementedError

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
        raise NotImplementedError

    def build_sep_optimizer(self):
        raise NotImplementedError

    def default_feed_dict(self):
        return {self.coeff_kl: 1., self.coeff_ll: 1.}
