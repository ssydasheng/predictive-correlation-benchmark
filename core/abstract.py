import tensorflow as tf
import tensorflow_probability as tfp

class Abstract(object):
    def __init__(self, input_dim, float_type=tf.float32):
        self.float_type = float_type
        self.input_dim = input_dim

    def build_inputs(self):
        float_type = self.float_type
        self.x = tf.placeholder(float_type, shape=[None, self.input_dim], name='x')
        self.x_pred = tf.placeholder(float_type, shape=[None, self.input_dim], name='x_pred')
        self.y = tf.placeholder(float_type, shape=[None], name='y')
        self.y_pred = tf.placeholder(float_type, shape=[None], name='y_pred')
        self.n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
        self.learning_rate_ph = tf.placeholder(float_type, shape=[], name='learning_rate')
        self.m = tf.placeholder(tf.int32, shape=[], name='mini_batch_size')

        # self.x1 = tf.placeholder(float_type, shape=[None, self.input_dim], name='x1')
        # self.x2 = tf.placeholder(float_type, shape=[None, self.input_dim], name='x2')

    def build_evaluation(self):
        self.eval_rmse = tf.sqrt(tf.reduce_mean((self.func_x_pred_mean - self.y_pred) ** 2))
        y_dist = tf.distributions.Normal(self.func_x_pred_mean, tf.sqrt(self.func_x_pred_std**2. + self.obs_var))
        self.eval_lld = tf.reduce_mean(y_dist.log_prob(self.y_pred))

        # joint lld on cpu
        # with tf.device('/cpu:0'):
        n = tf.shape(self.x_pred)[0]
        cov = tf.reshape(self.func_x_pred_cov, [n, n])
        cov = tf.linalg.set_diag(cov, tf.linalg.diag_part(cov) + tf.cast(self.obs_var, dtype=cov.dtype))
        loc = tf.cast(self.func_x_pred_mean, dtype=cov.dtype)
        self.y_x_pred_cov = cov
        ys_dist = tfp.distributions.MultivariateNormalFullCovariance(loc, cov,  allow_nan_stats=False,
                                                                     name="joint_lld")
        self.eval_joint_lld = ys_dist.log_prob(tf.cast(self.y_pred, dtype=cov.dtype))

    def build_function(self):
        raise NotImplementedError

    def build_optimizer(self):
        raise NotImplementedError

    def default_feed_dict(self):
        return {}