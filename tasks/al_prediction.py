### This file implements the prediction model for the selected points.
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np
import core.gpflowSlim as gfs
import json

from utils.common_utils import (regression_train,
                                regression_test)
from utils.model_utils import get_model
from easydict import EasyDict as edict
from utils.al_utils import HParams
from utils.logging import get_logger
from global_settings.constants import RESULT_AL_PATH, LOG_AL_PREDICTION_PATH, RESULT_AL_SELECTION_PATH


def load_cached_data(args, seed, itr, has_pool=False):
    base_method = 'gp' if args.base_method == 'svgp' else args.base_method
    path = osp.join(RESULT_AL_SELECTION_PATH,
                     args.criteria,
                     '%s_seed%d' % (args.dataset, seed),
                     '%s_itr%d.npy' % (base_method, itr))
    data = np.load(path, allow_pickle=True)
    if not has_pool:
        return HParams(train_x=data.item().get('train_x'), train_y=data.item().get('train_y'),
                       test_x=data.item().get('test_x'), test_y=data.item().get('test_y'))
    else:
        return HParams(train_x=data.item().get('train_x'), train_y=data.item().get('train_y'),
                       test_x=data.item().get('test_x'), test_y=data.item().get('test_y'),
                       pool_x=data.item().get('pool_x'), pool_y=data.item().get('pool_y'))


def _run(args, seed, print):
    test_rmses, test_llds = [], []
    BSZ, NBS = args.batch_size, args.n_base
    for i_point in range(0, args.active_iterations + 1):
        tf.reset_default_graph()
        raw_dataset = load_cached_data(args, seed, i_point)
        y_std = np.std(raw_dataset.train_y, 0, keepdims=True)
        y_std[y_std == 0] = 1
        y_mean = np.mean(raw_dataset.train_y, 0, keepdims=True)

        train_x, test_x = raw_dataset.train_x, raw_dataset.test_x
        train_y, test_y = raw_dataset.train_y, raw_dataset.test_y
        train_y, test_y = (train_y - y_mean) / y_std, (test_y - y_mean) / y_std
        optimizer_dataset = edict({
            'train_x': train_x, 'train_y': train_y,
            'test_x': test_x, 'test_y': test_y,
            'std_y_train': y_std,
        })
        N, input_dim = train_x.shape
        args.batch_size, args.n_base = min(BSZ, N), min(NBS, BSZ, N - 1)

        model, print_values, train_op, obs_var, _, _ = get_model(
            args, train_x, train_y, test_x, None, input_dim, print)
        global_step = model.global_step
        train_summary = tf.no_op()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        test_rmse, test_lld = None, None
        for epoch in range(1, args.epochs+1):
            regression_train(args, model, train_op, optimizer_dataset, epoch, print_values,
                             global_step, train_summary, None, sess, logger=print)
            if epoch % args.test_interval == 0 or epoch == args.epochs:
                test_rmse, test_lld = regression_test(args, model, optimizer_dataset, epoch, sess, obs_var, global_step,
                                                      None, seed, logger=print)

        test_rmses.append(test_rmse[0])
        test_llds.append(test_lld[0])

    return test_rmses, test_llds


def run(args):
    method = 'gp' if args.method == 'svgp' else args.method
    base_method = 'gp' if args.base_method == 'svgp' else args.base_method
    log_dir = osp.join(LOG_AL_PREDICTION_PATH, '%s_%s_%s'%(args.dataset, method, base_method))
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

    result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json'%(args.dataset, method, base_method))
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
