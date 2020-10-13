import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np
import json
from tensorflow.core.protobuf import rewriter_config_pb2
import copy

import core.gpflowSlim as gfs
from utils.model_utils import get_model
from utils.common_utils import (regression_test,
                                regression_train)
from utils.logging import get_logger, makedirs
from utils.al_utils import (get_selected_data_idxs,
                            fetch_active_learning_dataset,
                            normalize_active_learning_data,
                            update_dataset)
from global_settings.constants import RESULT_AL_PATH, LOG_AL_SELECTION_PATH, RESULT_AL_SELECTION_PATH

def _run(args, seed, print):
    # ================================== load and normalize data ==================================
    raw_dataset = fetch_active_learning_dataset(args, seed)
    if not osp.exists(osp.join(RESULT_AL_SELECTION_PATH, args.criteria, '%s_seed%d' % (args.dataset, seed))):
        makedirs(osp.join(RESULT_AL_SELECTION_PATH, args.criteria, '%s_seed%d' % (args.dataset, seed), 'a.py'))
    np.save(osp.join(RESULT_AL_SELECTION_PATH,
                     args.criteria,
                     '%s_seed%d' % (args.dataset, seed),
                     '%s_itr%d.npy' % (args.method, 0)),
            {'test_x': raw_dataset.test_x,
             'test_y': raw_dataset.test_y,
             'train_x': raw_dataset.train_x,
             'train_y': raw_dataset.train_y,
             'pool_x': raw_dataset.pool_x,
             'pool_y': raw_dataset.pool_y})
    print("Dataset: %s, train_r: %.2f, test_r: %.2f" % (args.dataset, args.train_ratio, args.test_ratio))

    test_rmses, test_llds = [], []
    BSZ, NBS = args.batch_size, args.n_base
    for i_point in range(1, args.active_iterations + 2):
        tf.reset_default_graph()
        dataset = normalize_active_learning_data(copy.deepcopy(raw_dataset))
        train_x, test_x, pool_x = dataset.train_x, dataset.test_x, dataset.pool_x
        train_y = dataset.train_y
        N, input_dim = train_x.shape
        args.batch_size, args.n_base = min(BSZ, N), min(NBS, BSZ, N-1)
        # ================================== setup model ==================================
        model, print_values, train_op, obs_var, corr_op, covar_op = get_model(args, train_x, train_y,
                                                                              test_x, pool_x, input_dim, print)
        global_step = model.global_step
        train_summary = tf.no_op()

        # ==================================  training and testing ==================================
        if args.method in ['hmc', 'emhmc']:
            print("HMC reset the config proto of Session.\n" * 10)
            config_proto = tf.ConfigProto()
            off = rewriter_config_pb2.RewriterConfig.OFF
            config_proto.graph_options.rewrite_options.arithmetic_optimization = off
            sess = tf.Session(config=config_proto)
        else:
            sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        test_rmse, test_lld = None, None
        learning_rate = args.learning_rate
        for epoch in range(1, args.epochs+1):
            if epoch % args.lr_ann_epochs == 0:
                learning_rate = learning_rate * args.lr_ann_ratio
            regression_train(args, model, train_op, dataset, epoch, print_values, global_step,
                             train_summary, None, sess, learning_rate=learning_rate, logger=print)
            if epoch == args.epochs:
                test_rmse, test_lld = regression_test(args, model, dataset, epoch, sess, obs_var, global_step,
                                                      None, seed, print)
        test_rmses.append(test_rmse[0])
        test_llds.append(test_lld[0])

        # ================== select the new data point(s). ===========================================
        data_idxs = get_selected_data_idxs(train_x, test_x, pool_x, dataset, args, covar_op, corr_op,
                                           model, sess, n_particles=args.eval_cov_samples)
        raw_dataset = update_dataset(raw_dataset, data_idxs)
        np.save(osp.join(RESULT_AL_SELECTION_PATH,
                         args.criteria,
                         '%s_seed%d' % (args.dataset, seed),
                         '%s_itr%d.npy' % (args.method, i_point)),
                {'test_x': raw_dataset.test_x,
                 'test_y': raw_dataset.test_y,
                 'train_x': raw_dataset.train_x,
                 'train_y': raw_dataset.train_y})
        print("Iteration %d done. Selected data point %d/%d. (%d/%d/%d)" % (i_point,
                                                                            i_point,
                                                                            args.active_iterations,
                                                                            raw_dataset.train_x.shape[0],
                                                                            raw_dataset.pool_x.shape[0],
                                                                            raw_dataset.test_x.shape[0]))
        sess.close()

    return test_rmses, test_llds


def run(args):
    method = 'gp' if args.method == 'svgp' else args.method
    log_dir = osp.join(LOG_AL_SELECTION_PATH, '%s_%s'%(args.dataset, method))
    logger = get_logger(args.dataset, log_dir, __file__)
    print = logger.info

    # set jitter
    if args.dataset in ['concrete']:
        gfs.settings.set_jitter(3e-5)
    if args.dataset in ['naval']:
        gfs.settings.set_jitter(1e-4)

    rmse_results, lld_results = [], []
    for seed in range(args.init_seed, args.n_runs+args.init_seed):
        rmse, ll = _run(args, seed, print)
        rmse_results.append(rmse)
        lld_results.append(ll)

    rmse_results = np.array(rmse_results)
    lld_results = np.array(lld_results)

    rmse_mean = np.mean(rmse_results, axis=0)
    rmse_var = np.var(rmse_results, axis=0)
    lld_mean = np.mean(lld_results, axis=0)
    lld_var = np.var(lld_results, axis=0)
    print("rmse_mean: %s" % rmse_mean.tolist())
    print("rmse_var: %s" % rmse_var.tolist())

    print("lld_mean: %s" % lld_mean.tolist())
    print("lld_var: %s" % lld_var.tolist())

    result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json'%(args.dataset, method, method))
    with open(result_dir, 'w') as f:
        res = {
            'dataset': args.dataset,
            'criteria': args.criteria,
            'rmse_mean': rmse_mean.tolist(),
            'rmse_var': rmse_var.tolist(),
            'lld_mean': lld_mean.tolist(),
            'lld_var': lld_var.tolist(),
            'rmse': rmse_results.tolist(),
            'lld': lld_results.tolist(),
        }
        json.dump(res, f)
