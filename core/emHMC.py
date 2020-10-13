import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from core.abstractVI import AbstractVI
from tensorflow.python.framework import tensor_shape


class EMHMC(AbstractVI):

    def __init__(self, posterior, obs_var, input_dim, N, layer_sizes, n_samples=100, n_chains=1, n_burnin=10000,
                n_steps_between_results=50, step_size=0.1, mini_particles=100, leap_frog_steps=5,
                n_steps_per_em=1, n_steps_adapt_step_size=1, eta=1.):
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.n_burnin = n_burnin
        self.n_steps_between_results = n_steps_between_results
        self.step_size = step_size
        self.n_steps_per_em = n_steps_per_em
        self.n_steps_adapt_step_size = n_steps_adapt_step_size

        self.layer_sizes = layer_sizes
        self.leap_frog_steps = leap_frog_steps
        super().__init__(posterior, obs_var, input_dim, N, mini_particles=mini_particles, eta=eta)

    def build_(self):
        self.build_inputs()
        self.build_coeff()
        self.build_prior_ws()
        self.build_states()
        self.build_optimizer()
        self.build_function()
        self.build_evaluation()

    def log_potential(self, *w_particles):
        nc = self.n_chains
        # [n_chains, shape_]
        ws = [w[:, :-1] for w in w_particles]
        bs = [w[:, -1:] for w in w_particles]

        # [n_chains, bs]
        func_x = tf.squeeze(self.posterior(self.x, ws, bs, no_sample_dim=False), -1)
        y_x_dist = tfp.distributions.Normal(func_x, self.obs_var[..., None] ** 0.5)

        y_obs = tf.tile(self.y[None], [nc, 1])
        log_likelihood = tf.reduce_mean(y_x_dist.log_prob(y_obs)) * self.N
        log_prior= tf.add_n([tf.reduce_sum(pw.log_prob(w))
                             for w, pw in zip(w_particles, self.prior_ws)]) / nc
        return self.n_chains * (log_likelihood + log_prior)

    def build_states(self):
        self._state_step_size = tf.Variable(self.step_size, trainable=False, name='step_size')
        self._state_w_particles = [tf.Variable(iw, trainable=False) for iw in self._init_particles]
        self.accumulated = [tf.Variable(tf.zeros([self.n_samples]+list(iw.shape), dtype=tf.float32), trainable=False)
                            for iw in self._init_particles]
        self._n_acc = tf.Variable(0, trainable=False)

    @property
    def ws(self):
        return [tf.reshape(w[:self._n_acc], [-1]+w.get_shape().as_list()[-2:])[:, :-1] for w in self.accumulated]

    @property
    def bs(self):
        return [tf.reshape(w[:self._n_acc], [-1]+w.get_shape().as_list()[-2:])[:, -1:] for w in self.accumulated]

    # @property
    # def ws(self):
    #     return [w[:, :-1] for w in self._state_w_particles]
    #
    # @property
    # def bs(self):
    #     return [w[:, -1:] for w in self._state_w_particles]

    def build_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=self.log_potential,
                num_leapfrog_steps=self.leap_frog_steps,
                step_size=self._state_step_size,
                state_gradients_are_stopped=True,
            ),
            num_adaptation_steps=self.n_steps_adapt_step_size)

        def trace_fn(_, pkr):
            return (pkr.inner_results.log_accept_ratio,
                    pkr.inner_results.accepted_results.step_size)

        samples, (avg_log_acc, avg_step) = tfp.mcmc.sample_chain(
            num_results=self.n_steps_per_em,
            num_burnin_steps=0,
            num_steps_between_results=1,
            current_state=self._state_w_particles,
            kernel=self.adaptive_hmc,
            trace_fn=trace_fn)

        self.update_step_size_op = self._state_step_size.assign(avg_step[-1])
        self.update_ws_op = [var.assign(s[-1]) for var, s in zip(self._state_w_particles, samples)]
        self.infer_latent = tf.group(self.update_ws_op + [self.update_step_size_op])

        # infer hyper-parameter
        self.hyper_optimizer = tf.train.AdamOptimizer(self.learning_rate_ph)
        flag = len(self.params_likelihood) or len(self.params_prior)
        params = self.params_likelihood + self.params_prior
        with tf.control_dependencies([self.infer_latent]):
            self.target_log_prob = self.log_potential(*self._state_w_particles) / (self.N * self.n_chains)
            self.infer_hyper = self.hyper_optimizer.minimize(-self.target_log_prob, var_list=params)\
                if flag else tf.no_op()

        ## update accumated_samples
        def update_func():
            ops = []
            for w, s in zip(self.accumulated, samples):
                new_acc = tf.concat([w[:self._n_acc], s[-1:], w[self._n_acc+1:]], 0)
                ops.append(w.assign(new_acc))
            with tf.control_dependencies(ops):
                with tf.control_dependencies([self._n_acc.assign(self._n_acc + 1)]):
                    return tf.constant(0.)

        flag1 = self.global_step > self.n_burnin
        flag2 = tf.equal(tf.mod(self.global_step, self.n_steps_between_results), 0)
        self.update_acc_op = tf.cond(
            tf.logical_and(flag1, flag2),
            update_func,
            lambda: tf.constant(0.)
        )
        ### update global step
        with tf.control_dependencies([self.update_acc_op]):
            self.update_gs_op = self.global_step.assign(self.global_step + 1)

        self.infer_joint = tf.group(self.infer_latent, self.infer_hyper, self.update_gs_op, self.update_acc_op)

    def build_function(self):
        self.func_x_pred = self.posterior(self.x_pred, self.ws, self.bs, no_sample_dim=False)
        self.func_x_pred = tf.squeeze(self.func_x_pred, 2)
        self.func_x_pred_mean, self.func_x_pred_var = tf.nn.moments(self.func_x_pred, 0)
        self.func_x_pred_std = tf.sqrt(self.func_x_pred_var)

        # ======== add covariance computation =========
        self.func_x_pred_cov = tfp.stats.covariance(self.func_x_pred, sample_axis=0, event_axis=-1)

    def build_prior_ws(self):
        self.prior_ws = []
        self._init_particles = []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):

            mean_w = tf.zeros(shape=[n_in+1, n_out], dtype=tf.float32)
            var_w = tf.ones([self.n_chains, n_in+1, n_out], dtype=tf.float32) * self.eta
            self.prior_ws.append(tfp.distributions.Normal(mean_w, tf.sqrt(var_w)))

            init_ws = np.random.normal(size=[self.n_chains, n_in+1, n_out]).astype('float32') # * self.eta**0.5
            self._init_particles.append(init_ws)


class EMHMCScale(EMHMC):

    def build_(self):
        self.build_inputs()
        self.build_coeff()
        self.build_prior_ws()
        self.build_function()
        self.build_evaluation()

    def compute_func_x_pred_with_loop(self):
        raise NotImplementedError