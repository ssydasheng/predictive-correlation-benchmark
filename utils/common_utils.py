import argparse
import numpy as np
import tensorflow as tf
import logging
import time
import json
import os
import os.path as osp
import core.gpflowSlim as gfs
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from easydict import EasyDict as edict
from scipy.stats import pearsonr as pr
from scipy.stats import spearmanr as spr
from scipy.interpolate import interpn
mpl.use('Agg')
sns.set_style("whitegrid")


def set_jitter(args):
    if args.dataset in ['concrete']:
        gfs.settings.set_jitter(3e-5)
    if args.dataset in ['naval']:
        gfs.settings.set_jitter(1e-4)
    if args.dataset in ['year'] and args.gpnet == 'deep':
        gfs.settings.set_jitter(3e-4)


def regression_train(args,
                     model,
                     train_op,
                     data, epoch,
                     print_values,
                     global_step,
                     train_summary,
                     summary_writer,
                     sess,
                     learning_rate=None,
                     logger=None,
                     train_obs=True):
    if train_op is None:
        return None
    if logger is None:
        logger = print
    train_x, train_y = data.train_x, data.train_y
    N = train_x.shape[0]
    BSZ = args.batch_size
    if BSZ <= 0:
        BSZ = N
    epoch_iters = max(N // BSZ, 1)
    # for epoch in range(1, args.epochs + 1):
    indices = np.random.permutation(N)
    train_x, train_y = train_x[indices], train_y[indices]
    default_feed_dict = model.default_feed_dict()

    begin_time = time.time()
    for iter in range(epoch_iters):
        if args.method == 'ensemble':
            n_networks = args.num_networks
            indices = [np.random.choice(train_x.shape[0], BSZ) for _ in range(n_networks)]
            x_batch = np.concatenate([train_x[indices[_]] for _ in range(n_networks)], 0)
            y_batch = np.concatenate([train_y[indices[_]] for _ in range(n_networks)], 0)
        else:
            x_batch = train_x[iter * BSZ: (iter + 1) * BSZ]
            y_batch = train_y[iter * BSZ: (iter + 1) * BSZ]
        lr = args.learning_rate if learning_rate is None else learning_rate
        feed_dict = {model.x: x_batch, model.y: y_batch, model.learning_rate_ph: lr,
                     model.n_particles: args.train_samples, model.m: x_batch.shape[0]}
        feed_dict.update(default_feed_dict)

        names, values = zip(*print_values.items())
        names, values = list(names), list(values)
        res = sess.run([train_op] + values + [global_step, train_summary], feed_dict=feed_dict)

        if iter == epoch_iters - 1 and epoch % args.print_interval == 0:
            if summary_writer is not None:
                summary_writer.add_summary(res[-1], global_step=res[-2])
            format_ = ' | '.join(['{}={:.5f}'.format(n, l) for n, l in zip(names, res[1:-2])])
            elapsed_time = time.time() - begin_time
            logger('>>> Epoch {:5d}/{:5d} [{:4f}s] | lr={:5f} | '.format(epoch, args.epochs, elapsed_time, lr) + format_)

def regression_test(args,
                    model,
                    data,
                    epoch,
                    sess,
                    obs_var,
                    global_step,
                    summary_writer,
                    seed,
                    logger=None):
    if logger is None:
        logger = print
    test_x, test_y, std_y_train = data.test_x, data.test_y, data.std_y_train
    feed_dict = {model.x: data.train_x, model.y: data.train_y, model.x_pred: test_x, model.y_pred: test_y,
                 model.n_particles: args.test_samples}
    rmse, lld, ov, gs = sess.run([model.eval_rmse, model.eval_lld, obs_var, global_step],
                                 feed_dict=feed_dict)
    rmse, lld = rmse * std_y_train, lld - np.log(std_y_train)

    if summary_writer is not None:
        summary = tf.Summary()
        summary.value.add(tag="test/rmse", simple_value=rmse)
        summary.value.add(tag="test/lld", simple_value=lld)
        summary_writer.add_summary(summary, gs)
    logger('>>> TEST: Seed {:5d} >>> Epoch {:5d}/{:5d} | rmse={:.5f} | lld={:.5f} | obs_var={:.5f}'.format(
        seed, epoch, args.epochs, rmse[0], lld[0], np.mean(ov)))

    return rmse, lld


# ================================== for compute pairwise lld ===========================================
def normal_log_lld(mu, var, y):
    exp_out_part = 1 / (2 * np.pi * var) ** 0.5
    exp_part = np.exp(-0.5 * (y - mu) ** 2 / var)
    exp_out_part = np.squeeze(exp_out_part)
    exp_part = np.squeeze(exp_part)
    ld = np.multiply(exp_out_part, exp_part)

    lld = np.log(ld)
    return lld


def compute_pair_wise_lld(y, mu, cov, ids, ov, std_y_train, homo_noise=True):
    x1_ids = ids[:, 0]
    x2_ids = ids[:, 1]

    y1, y2 = y[x1_ids], y[x2_ids]
    mu1, mu2 = mu[x1_ids], mu[x2_ids]
    var1, var2 = cov[x1_ids, x1_ids], cov[x2_ids, x2_ids]

    if not homo_noise:
        ov_x1 = ov[x1_ids]
        ov_x2 = ov[x2_ids]
    else:
        ov_x1 = ov
        ov_x2 = ov
    x1_lld = normal_log_lld(mu1, var1 + ov_x1, y1)

    pre_cond = np.multiply(cov[x1_ids, x2_ids], 1.0 / (var1 + ov_x1))
    mu2_cond_x1 = mu2 - np.multiply(pre_cond, y1 - mu1)
    var2_cond_x1 = (var2 + ov_x2) - np.multiply(pre_cond, cov[x1_ids, x2_ids])

    x2_lld_cond_x1 = normal_log_lld(mu2_cond_x1, var2_cond_x1, y2)

    pair_lld = x1_lld + x2_lld_cond_x1
    pair_lld = np.mean(pair_lld) - 2 * np.log(std_y_train)
    # import pdb; pdb.set_trace()
    return pair_lld


# ============================= for compute correlations and visualization =============================
def plot_scatter(oracle_x, model_x, n_pool, n_test, name, method, str_, pearson='Pearson', **kwargs):
    assert oracle_x.shape[0] == model_x.shape[0]
    assert oracle_x.shape[0] == n_pool + n_test
    # x is the oracle, y is the model
    plt.clf()
    # sort according to oracle
    name = name + '-diag'
    x, y = np.diag(oracle_x)[:n_pool].tolist(), np.diag(model_x)[:n_pool].tolist()
    x, y = zip(*sorted(zip(x, y)))
    data, x_e, y_e = np.histogram2d(x, y, bins=[40, 40])
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([np.array(x), np.array(y)]).T, method="splinef2d",
                bounds_error=False)
    plt.scatter(x, y, c=z, s=2, **kwargs)
    plt.title('%s_%s_%s' % (pearson, name, method))
    plt.xlabel('Oracle')
    plt.ylabel(method)
    plt.tight_layout()
    # plt.show()
    plt.savefig(osp.join('results', str_, 'figs/', '%s_%s_%s.png' % (pearson, name, method)))
    file_name = osp.join('results', str_, 'figs/', '%s_%s_%s.json' % (pearson, name, method))
    with open(file_name, 'w') as f:
        json.dump({'x': x, 'y': y}, f)

    plt.clf()
    # sort according to oracle
    name = name + '-offdiag'
    x, y = np.reshape(oracle_x[:n_pool][:, n_pool:], -1).tolist(), np.reshape(model_x[:n_pool][:, n_pool:], -1).tolist()
    x, y = zip(*sorted(zip(x, y)))
    data, x_e, y_e = np.histogram2d(x, y, bins=[40, 40])
    z = interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([np.array(x), np.array(y)]).T,
                method="splinef2d",
                bounds_error=False)
    plt.scatter(x, y, c=z, s=2, **kwargs)
    plt.title('%s_%s_%s' % (pearson, name, method))
    plt.xlabel('Oracle')
    plt.ylabel(method)
    plt.tight_layout()
    # plt.show()
    plt.savefig(osp.join('results', str_, 'figs/', '%s_%s_%s.png' % (pearson, name, method)))
    file_name = osp.join('results', str_, 'figs/', '%s_%s_%s.json' % (pearson, name, method))
    with open(file_name, 'w') as f:
        json.dump({'x': x, 'y': y}, f)


def compute_pearson_and_spearman_r(A, B, n_pool, n_test):
    assert A.shape[0] == n_pool + n_test
    A_diag = np.diag(A)[:n_pool].tolist()
    B_diag = np.diag(B)[:n_pool].tolist()

    A_pool_test = A[:n_pool][:, n_pool:]
    B_pool_test = B[:n_pool][:, n_pool:]
    A_offdiag = np.reshape(A_pool_test, -1).tolist()
    B_offdiag = np.reshape(B_pool_test, -1).tolist()

    pr_diag, pr_diag_p = pr(A_diag, B_diag)
    pr_offdiag, pr_offdiag_p = pr(A_offdiag, B_offdiag)

    spr_diag, spr_diag_p = spr(A_diag, B_diag)
    spr_offdiag, spr_offdiag_p = spr(A_offdiag, B_offdiag)
    return pr_diag, pr_offdiag, spr_diag, spr_offdiag, pr_diag_p, pr_offdiag_p, spr_diag_p, spr_offdiag_p


def reuse_variables(scope):
    """
    A decorator for transparent reuse of tensorflow
    `Variables <https://www.tensorflow.org/api_docs/python/tf/Variable>`_ in a
    function. The decorated function will automatically create variables the
    first time they are called and reuse them thereafter.
    .. note::
        This decorator is internally implemented by tensorflow's
        :func:`make_template` function. See `its doc
        <https://www.tensorflow.org/api_docs/python/tf/make_template>`_
        for requirements on the target function.
    :param scope: A string. The scope name passed to tensorflow
        `variable_scope()
        <https://www.tensorflow.org/api_docs/python/tf/variable_scope>`_.
    """
    return lambda f: tf.make_template(scope, f)


def reuse(scope):
    """
    (Deprecated) Alias of :func:`reuse_variables`.
    """
    warnings.warn(
        "The `reuse()` function has been renamed to `reuse_variables()`, "
        "`reuse()` will be removed in the coming version (0.4.1)",
        FutureWarning)
    return reuse_variables(scope)


# ================== for configs ============================
def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = edict(config_dict)

    return config, config_dict


def process_config(json_file):
    """Process a json file into a config file.
    Where we can access the value using .xxx
    Note: we will need to create a similar directory as the config file.
    """
    config, _ = get_config_from_json(json_file)
    paths = json_file.split('/')[1:-1]
    summary_dir = ["./results"] + paths + [config.exp_name, "summary/"]
    ckpt_dir = ["./results"] + paths + [config.exp_name, "checkpoint/"]
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    return config


def get_logger(name, logpath, filepath, package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name + time.strftime("-%Y%m%d-%H%M%S")
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    logger.info(filepath)
    with open(filepath, 'r') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    return logger


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = process_config(args.config)

    return config


def init_logger(config, main_file):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=main_file, package_files=[])
    logger.info(dict(config))

    return logger


