import os
import os.path as osp
import sys
root_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(root_path)

import numpy as np
import scipy
import tensorflow as tf
from easydict import EasyDict as edict

from scipy.stats import pearsonr as compute_pearsonr
from tensorflow.core.protobuf import rewriter_config_pb2
from utils.common_utils import (regression_train,
                                compute_pearson_and_spearman_r)
from utils.logging import get_logger, makedirs
from utils.model_utils import get_model
from utils.al_utils import get_selected_data_idxs, update_dataset
from global_settings.constants import TOY_DATA_PATH, RESULT_TOY_METRIC_PATH, LOG_TOY_METRIC_PATH
from utils.toy_utils import compute_cov_relu_kernel


def gen_f_correlation(model):
    cov = model.func_x_pred_cov  # N * N
    diag_cov_sqrt = tf.math.sqrt(tf.linalg.tensor_diag_part(cov))
    corr = cov * tf.reshape(1. / diag_cov_sqrt, [-1, 1]) * tf.reshape(1. / diag_cov_sqrt, [1, -1])
    return corr, cov


def compute_oracle_lld_rmse(dataset):
    tr_X, te_X = dataset.train_x, dataset.test_x
    tr_Y, te_Y = dataset.train_y, dataset.test_y
    n_tr, n_te = tr_X.shape[0], te_X.shape[0]
    X = np.concatenate([tr_X, te_X], 0)
    KXX = compute_cov_relu_kernel(X)
    Ktrtr_inv = np.linalg.inv(KXX[:n_tr, :][:, :n_tr] + dataset.obs_var * np.eye(n_tr))
    Ktetr = KXX[n_tr:, :][:, :n_tr]
    Ktete = KXX[n_tr:, :][:, n_tr:]
    Ktete_post = Ktete - Ktetr @ Ktrtr_inv @ Ktetr.T + dataset.obs_var * np.eye(n_te)
    pred_mu = Ktetr @ Ktrtr_inv @ tr_Y
    test_lld = np.mean(np.log(scipy.stats.norm(pred_mu, np.diag(Ktete_post) ** 0.5).pdf(te_Y) + 1e-80))
    test_rmse = np.mean((te_Y - pred_mu) ** 2) ** 0.5
    return test_lld, test_rmse


def _run(args, seed, print):
    tf.reset_default_graph()
    dataset = edict(dict(np.load(os.path.join(
        root_path, TOY_DATA_PATH, 'reluLarge', 'dim%d_seed%s.npz' % (args.input_dim, seed)))))

    # ================================== setup model ==================================
    BSZ, NBS = args.batch_size, args.n_base
    train_x, test_x, pool_x = dataset.train_x, dataset.test_x, dataset.pool_x
    train_y = dataset.train_y
    N, input_dim = train_x.shape
    args.batch_size, args.n_base = min(BSZ, N), min(NBS, BSZ, N - 1)
    given_obs_var = dataset.obs_var
    model, print_values, train_op, obs_var, corr_op, covar_op = get_model(
        args, train_x, train_y, test_x, pool_x, input_dim, print,
        mini_particles=1, given_obs_var=given_obs_var)
    corr_f_op, covar_f_op = gen_f_correlation(model)
    train_summary = tf.no_op()
    global_step = model.global_step

    # ==================================  training ==================================
    if args.method in ['hmc', 'emhmc']:
        print("HMC reset the config proto of Session.\n" * 10)
        config_proto = tf.ConfigProto()
        off = rewriter_config_pb2.RewriterConfig.OFF
        config_proto.graph_options.rewrite_options.arithmetic_optimization = off
        sess = tf.Session(config=config_proto)
    else:
        sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    learning_rate = args.learning_rate

    for epoch in range(1, args.epochs + 1):
        if epoch % args.lr_ann_epochs == 0:
            learning_rate = learning_rate * args.lr_ann_ratio
        regression_train(args, model, train_op, dataset, epoch, print_values, global_step,
                        train_summary, None, sess, learning_rate=learning_rate, logger=print)


    # ==================================  testing ==================================
    pool_test_x = np.concatenate([dataset.pool_x, dataset.test_x], 0)
    feed_dict = {
        model.x_pred: pool_test_x, model.x: dataset.train_x, model.y: dataset.train_y,
        model.n_particles: args.eval_cov_samples
    }
    while True:
        # loop until the results are not nan.
        covariance_f, correlation_f, model_mu, model_var, model_obs_var = sess.run(
            [covar_f_op, corr_f_op, model.func_x_pred_mean, model.func_x_pred_var, model.obs_var],
            feed_dict=feed_dict)

        if not (np.isnan(covariance_f).any() or np.isnan(correlation_f).any()
                or np.isnan(model_mu).any() or np.isnan(model_var).any()):
            break

    # ========== compute Pearson and Spearman correlations with respect to the Oracle ============

    Kff = dataset.posterior_Kff
    reci_sqrt_diag_Kff = 1.0 / np.diag(Kff) ** 0.5
    gt_corr_f = Kff * (np.reshape(reci_sqrt_diag_Kff, [-1, 1]) @ np.reshape(reci_sqrt_diag_Kff, [1, -1]))
    var_pr_f, _, var_spr_f, _, _, _, _, _ = compute_pearson_and_spearman_r(covariance_f,
                                                                           Kff,
                                                                           pool_x.shape[0],
                                                                           test_x.shape[0])

    _, corr_pr_f, _, corr_spr_f, _, _, _, _ = compute_pearson_and_spearman_r(correlation_f,
                                                                             gt_corr_f,
                                                                             pool_x.shape[0],
                                                                             test_x.shape[0])

    print('*' * 20)
    print('[F][Var] Pearson: %.4f, Spearman: %.4f' % (var_pr_f, var_spr_f))
    print('[F][Cor] Pearson: %.4f, Spearman: %.4f' % (corr_pr_f, corr_spr_f))

    # ======== select the new data points (BMIG, MIG, TIG) and compute the Test LLD afterwards . ============
    org_crit = args.criteria

    args.criteria = 'BatchMIG'
    BMIG_data_idxs = get_selected_data_idxs(train_x, test_x, pool_x,
                                            dataset, args, covar_op, corr_op,
                                            model, sess, n_particles=args.eval_cov_samples)
    BMIG_raw_dataset = update_dataset(dataset, BMIG_data_idxs)
    BMIG_raw_dataset.obs_var = given_obs_var

    args.criteria = 'mig'
    MIG_data_idxs = get_selected_data_idxs(train_x, test_x, pool_x,
                                            dataset, args, covar_op, corr_op,
                                            model, sess, n_particles=args.eval_cov_samples)
    MIG_raw_dataset = update_dataset(dataset, MIG_data_idxs)
    MIG_raw_dataset.obs_var = given_obs_var

    args.criteria = 'tig'
    TIG_data_idxs = get_selected_data_idxs(train_x, test_x, pool_x,
                                           dataset, args, covar_op, corr_op,
                                           model, sess, n_particles=args.eval_cov_samples)
    TIG_raw_dataset = update_dataset(dataset, TIG_data_idxs)
    TIG_raw_dataset.obs_var = given_obs_var

    args.criteria = org_crit

    BMIG_lld, BMIG_rmse = compute_oracle_lld_rmse(BMIG_raw_dataset)
    MIG_lld, MIG_rmse = compute_oracle_lld_rmse(MIG_raw_dataset)
    TIG_lld, TIG_rmse = compute_oracle_lld_rmse(TIG_raw_dataset)
    print('>>> BatchMIG RMSE={:5f}, LLD={:5f}. | MIG  RMSE={:5f}, LLD={:5f}. | TIG  RMSE={:5f}, LLD={:5f}.'.format(
        BMIG_rmse, BMIG_lld, MIG_rmse, MIG_lld, TIG_rmse, TIG_lld))
    print('*' * 20)

    # =================== compute predictive covariances ===================
    pred_mean, pred_cov = sess.run(
        [model.func_x_pred_mean, model.y_x_pred_cov],
        feed_dict={ model.x_pred: test_x, model.x: train_x, model.y: train_y,
                    model.n_particles: args.eval_cov_samples}
    )

    # ====================================== save results ====================================== #
    method = 'gp' if args.method == 'svgp' else args.method
    np.savez(os.path.join(root_path, RESULT_TOY_METRIC_PATH, '%s_seed%d_%s.npz' % (method, seed, args.note)),
             **{'corr_pr_f': corr_pr_f,
                'corr_spr_f': corr_spr_f,
                'var_pr_f': var_pr_f,
                'var_spr_f': var_spr_f,
                'BMIG_test_rmse': BMIG_rmse,
                'BMIG_test_lld': BMIG_lld,
                'TIG_test_rmse': TIG_rmse,
                'TIG_test_lld': TIG_lld,
                'MIG_test_lld': MIG_lld,
                'MIG_test_rmse': MIG_rmse,
                'test_pred_mean': pred_mean,
                'test_pred_cov': pred_cov
                })


def run(args):
    log_dir = os.path.join(root_path, LOG_TOY_METRIC_PATH)
    logger = get_logger(args.dataset, log_dir, __file__)
    print = logger.info

    makedirs(osp.join(root_path, RESULT_TOY_METRIC_PATH, 'a.py'))
    for seed in range(1, 51):
        _run(args, seed, print)