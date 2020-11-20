import core.gpflowSlim as gfs
import numpy as np
import tensorflow as tf
from core.gaussianFVI import GaussianFVI, MiniBatchGaussianFVI
from core.gpr import SVGPGPR, FullGPR
from core.wvi import BBB
from core.ensemble import EnsembleNN
from core.homoscedasticEnsemble import HomoEnsembleNN
from core.dropout import Dropout
from core.noisy_kfac import NoisyKFAC
from core.emHMC import EMHMC
from utils.kernel_utils import NKNInfo
from utils.kernels import NNGPReLUKernel
from utils.nets import get_posterior, KFAC_MLP, mlp_input_outer
from utils.gpnet import get_gpnet
from utils.utils import get_kemans_init, median_distance_local, median_distance_global


# ======================== for building the model ==========================
def get_model(args,
              train_x,
              train_y,
              test_x,
              pool_x,
              input_dim,
              logger,
              dim_groups=None,
              oracle_N=None,
              mini_particles=100,
              given_obs_var=None):

    # TODO: A bit messy here, make it more readable

    N = train_x.shape[0]
    logger('** N: = %d\n' % N)
    with tf.compat.v1.variable_scope('prior'):
        if hasattr(args, 'kernel'):
            prior_kernel, _ = gen_kern(args, train_x, input_dim, dim_groups=dim_groups)
        else:
            prior_kernel = None

        if hasattr(args, 'opt_eta'):
            if args.opt_eta:
                log1e_eta = tf.get_variable('log1e_eta', dtype=tf.float32,
                                            initializer=float(np.log(np.exp(args.init_eta)-1)))
            else:
                log1e_eta = float(np.log(np.exp(args.init_eta)-1))
            eta = tf.nn.softplus(log1e_eta)
        else:
            eta = None

    if given_obs_var is None:
        with tf.variable_scope('likelihood'):
            obs_logstd = tf.get_variable('obs_logstd', shape=[], initializer=tf.constant_initializer(-2))
            obs_var = tf.exp(2. * obs_logstd)
    else:
        obs_var = tf.constant(given_obs_var, dtype=tf.float32)

    activation = tf.nn.relu
    if args.method == 'fbnn':
        with tf.variable_scope('posterior'):
            init_kern, ls = gen_kern(args, train_x, input_dim)
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden
        model = MiniBatchGaussianFVI(
            prior_kernel,
            get_gpnet(args.gpnet)('posterior', layer_sizes, mvn=False, kernel=[init_kern, ls], activation=activation),
            rand_generator=rand_generator(args, train_x, prior_kernel),
            obs_var=obs_var, input_dim=input_dim, n_rand=args.n_rand, N=N)
        print_values = {'elbo': model.elbo, 'kl': model.kl, 'logLL': model.log_likelihood, 'obs_var': obs_var}
        train_op = model.infer_joint
        # corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], 'gaussian')
    elif args.method == 'svgp':
        inducing_points = get_kemans_init(train_x, args.n_base)
        model = SVGPGPR(prior_kernel, gfs.likelihoods.Gaussian(0.1), Z=inducing_points,
                        input_dim=input_dim, N=train_x.shape[0] if oracle_N is None else oracle_N)
        train_op = model.infer_joint
        obs_var = model.obs_var
        print_values = {'Loss': model.loss, 'obs_var': model.obs_var}
        # corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], 'svgp')
    elif args.method == 'gp':
        if args.dataset in ['wine', 'energy']:
            min_var = 1e-8
        else:
            min_var = 1e-5
        model = FullGPR(train_x, np.expand_dims(train_y, 1), prior_kernel, input_dim, N, min_var=min_var)
        print_values = {'Loss': model.loss, 'obs_var': model.obs_var}
        train_op = model.infer_joint
        obs_var = model.obs_var

        if given_obs_var is not None:
            obs_logstd = model.GP.likelihood._variance.vf_val
            assign_obsvar_op = tf.assign(obs_logstd, tf.cast(tf.log(tf.exp(given_obs_var) - 1.), obs_logstd.dtype))
            train_op = tf.group(train_op, assign_obsvar_op)

        # corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], 'fullgp')
    elif args.method == 'dropout':
        lengthscale = 1e-2
        dropout_rate = args.dropout_rate
        if given_obs_var is None:
            tau = args.dropout_tau
        else:
            tau = given_obs_var
        reg = lengthscale ** 2 * (1 - dropout_rate) / (2. * N * tau)
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]
        model = Dropout(get_posterior('nn_relu')(layer_sizes,
                                                 dropout_rate=dropout_rate,
                                                 regularization=reg,
                                                 dropout_share_mask=True),
                        input_dim, N,
                        test_particles=args.eval_cov_samples,
                        obs_var=tau)
        train_op = model.infer_joint
        obs_var = model.obs_var
        print_values = {'Loss': model.loss, 'obs_var': model.obs_var}

        # corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], 'dropout')
    elif args.method == 'ensemble':
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [2]
        model = EnsembleNN([get_posterior('nn_relu')(layer_sizes, use_dropout=False) for _ in range(args.num_networks)], input_dim, N)
        print_values = {'Loss': model.loss}
        train_op = model.infer_joint
        obs_var = model.obs_var
        # corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], 'ensemble')
    elif args.method == 'homo_ensemble':
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]
        model = HomoEnsembleNN([get_posterior('nn_relu')(layer_sizes, use_dropout=False) for _ in range(args.num_networks)],
                                input_dim, N, obs_var)
        print_values = {'Loss': model.loss, 'obs_var': model.obs_var}
        train_op = model.infer_joint
        obs_var = model.obs_var
    elif args.method == 'bbb':
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]
        model = BBB(mlp_input_outer(tf.nn.relu)(layer_sizes, norm_fw=True), obs_var=obs_var,
                    input_dim=input_dim, N=N, layer_sizes=layer_sizes, eta=eta, mini_particles=mini_particles)
        print_values = {'elbo': model.elbo, 'kl': model.kl, 'logLL': model.log_likelihood,
                        'obs_var': obs_var, 'eta': eta}
        train_op = model.infer_joint
    elif args.method == 'nng':
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]
        model = NoisyKFAC(KFAC_MLP(activation)(layer_sizes, norm_fw=True), obs_var=obs_var, input_dim=input_dim,
                          N=train_x.shape[0], eta=eta, mini_particles=mini_particles)
        train_op = model.infer_joint
        obs_var = model.obs_var
        print_values = {'logLL': model.log_likelihood, 'elbo': model.elbo, 'obs_var': obs_var, 'eta': eta}
        # corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], 'nng')
    elif args.method == 'emhmc':
        assert args.epochs > args.hmc_burnin
        assert (args.epochs - args.hmc_burnin) < args.hmc_samples * args.num_steps_between_samples
        layer_sizes = [input_dim] + [args.n_units] * args.n_hidden + [1]
        model = EMHMC(mlp_input_outer(tf.nn.relu)(layer_sizes, norm_fw=True),
                      obs_var=obs_var, eta=eta, n_steps_per_em=args.hmc_n_steps_per_em,
                      n_steps_adapt_step_size=args.hmc_n_steps_adapt_step_size,
                      leap_frog_steps=args.hmc_leap_frog_steps,
                      step_size=args.hmc_step_size, input_dim=input_dim, n_chains=args.hmc_num_chains,
                      N=train_x.shape[0], layer_sizes=layer_sizes, n_burnin=args.hmc_burnin,
                      n_samples=args.hmc_samples, mini_particles=100,
                      n_steps_between_results=args.num_steps_between_samples)
        print_values = {'log_prob': model.target_log_prob, 'step_size': model._state_step_size,
                        'obs_var': obs_var, 'eta': eta}
        train_op = model.infer_joint
    else:
        raise NotImplementedError

    if pool_x is not None:
        corr_op, covar_op = gen_correlation(model, obs_var, pool_x.shape[0], test_x.shape[0], args.method, logger, args)
    else:
        corr_op, covar_op = None, None

    return model, print_values, train_op, obs_var, corr_op, covar_op


def rand_generator(args, train_x, prior_kernel):
    _rand = train_x
    rand = tf.data.Dataset.from_tensor_slices(tf.to_float(_rand)).shuffle(buffer_size=10000)
    rand = rand.batch(min(args.n_rand, _rand.shape[0]), drop_remainder=True).repeat()
    rand = rand.make_one_shot_iterator().get_next()
    rand = rand + tf.random_normal(tf.shape(rand)) * tf.to_float(prior_kernel.lengthscales/np.sqrt(2))
    def _f(*_args):
        return tf.stop_gradient(rand)
    return _f


def gen_correlation(model, obs_var, n_pool, n_test, method, logger, args):
    cov = model.func_x_pred_cov  # N * N
    masked_obsvar =  tf.cast(obs_var, dtype=cov.dtype) * tf.cast(
        tf.concat([[1.0] * n_pool, tf.cast([0.] * n_test, dtype=obs_var.dtype)], 0), dtype=cov.dtype)
    diag_cov = tf.linalg.tensor_diag_part(cov)
    sqrt_diag_cov = tf.math.sqrt(diag_cov + masked_obsvar)
    logger(sqrt_diag_cov)
    cov = cov + tf.diag(masked_obsvar)  # add observation variance
    corr = cov * tf.reshape(1./sqrt_diag_cov, [n_pool+n_test, 1]) * tf.reshape(1./sqrt_diag_cov, [1, n_pool+n_test])
    #TODO: previously we return corr**2, now we return corr.
    #TODO: what was the previous ensemble + total_info_gain used for ?
    return corr, cov


def gen_kern(args, train_x, input_dim, dim_groups=None):
    ls = median_distance_local(train_x)
    ls[abs(ls) < 1e-6] = 1.
    if args.kernel.lower() == 'rbf':
        if args.ARD:
            kernel = gfs.kernels.RBF(input_dim=input_dim, name='rbf', ARD=True, lengthscales=ls, min_ls=1e-4)
            return kernel, ls
        else:
            ls = median_distance_global(train_x)
            kernel = gfs.kernels.RBF(input_dim=input_dim, name='rbf', ARD=False, lengthscales=ls, min_ls=1e-4)
            return kernel, ls
    elif args.kernel.lower() == 'matern':
        print("Using Matern Kernel.\n" * 10)
        kernel = gfs.kernels.Matern52(input_dim=input_dim, name='matern', ARD=True, lengthscales=ls, min_ls=1e-5)
        return kernel, ls
    elif args.kernel.lower() == 'nkn':
        return NKNInfo(input_dim, ls), ls
    elif args.kernel.lower() == 'relu':
        return NNGPReLUKernel(input_dim, 1., 1., n_hiddens=1, ARD=args.ARD), ls

