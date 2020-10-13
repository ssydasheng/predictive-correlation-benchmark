import tensorflow as tf
import tensorflow_probability as tfp
from core.gpflowSlim.models import GPRFITC, SGPR, SVGP, GPR

from core.abstract import Abstract

class AbstractGPR(Abstract):
    """
    Base Class for Functional Variational Inference.

    :param posterior: the posterior network to be optimized.
    :param rand_generator: Generates measurement points.
    :param obs_var: Float. Observation variance.
    :param input_dim. Int.
    :param n_rand. Int. Number of random measurement points.
    """
    def __init__(self, prior_kernel, Z, input_dim, N):
        super().__init__(input_dim, float_type=tf.float64)

        self.prior_kernel = prior_kernel
        self._init_Z = Z
        self.N = N

        self.build_()

    def build_(self):
        self.build_inputs()
        self.build_gp()
        self.build_function()
        self.build_optimizer()
        self.build_evaluation()

    def build_gp(self):
        raise NotImplementedError

    @property
    def obs_var(self):
        return self.GP.likelihood.variance

    def build_function(self):
        self.func_x_pred_mean, self.func_x_pred_var = self.GP.predict_f(self.x_pred)
        self.func_x_pred_std = tf.squeeze(tf.sqrt(self.func_x_pred_var), 1)
        self.func_x_pred_mean = tf.squeeze(self.func_x_pred_mean, 1)
        with tf.device('/cpu:0'):
            _, func_x_pred_cov = self.GP.predict_f_full_cov(self.x_pred)
            self.func_x_pred_cov = tf.squeeze(func_x_pred_cov, 2)
        self.func_x_pred = self.GP.predict_f_samples(self.x_pred, self.n_particles)

    def build_optimizer(self):
        self.loss = self.GP.objective
        self.global_step = tf.Variable(0., trainable=False, name='global_step')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        self.infer_joint = self.optimizer.minimize(self.loss, global_step=self.global_step)




class FITCGPR(AbstractGPR):
    def __init__(self, prior_kernel, X, Y, Z, input_dim, N):
        self.X, self.Y = X, Y
        super(FITCGPR, self).__init__(prior_kernel, Z, input_dim, N)

    def build_gp(self):
        self.GP = GPRFITC(self.X, self.Y, kern=self.prior_kernel, Z=self._init_Z, name='fitc')


class VFEGPR(AbstractGPR):
    def __init__(self, prior_kernel, X, Y, Z, input_dim, N):
        self.X, self.Y = X, Y
        super(VFEGPR, self).__init__(prior_kernel, Z, input_dim, N)

    def build_gp(self):
        self.GP = SGPR(self.x, self.y[..., None], kern=self.prior_kernel, Z=self._init_Z, name='vfe')


class SVGPGPR(AbstractGPR):
    def __init__(self, prior_kernel, likelihood, Z, input_dim, N):
        self.likelihood = likelihood
        super(SVGPGPR, self).__init__(prior_kernel, Z, input_dim, N)

    def build_gp(self):
        self.GP = SVGP(self.x, self.y[..., None], kern=self.prior_kernel, Z=self._init_Z, likelihood=self.likelihood,
                       num_data=self.N, num_latent=1, name='svgp')

    def build_optimizer(self):
        super(SVGPGPR, self).build_optimizer()
        vars_ = tf.trainable_variables(self.GP.name)
        self.infer_only_Z = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=vars_)


class FullGPR(AbstractGPR):
    def __init__(self, X, Y, prior_kern, input_dim, N, min_var=1e-5):
        self.min_var = min_var
        super(FullGPR, self).__init__(prior_kern, None, input_dim, N)

    def build_gp(self):
        self.GP = GPR(self.x, tf.expand_dims(self.y, 1), num_latent=1, min_var=self.min_var, kern=self.prior_kernel, name='fullgp')
