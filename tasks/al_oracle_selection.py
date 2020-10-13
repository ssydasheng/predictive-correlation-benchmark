
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np
import json
import core.gpflowSlim as gfs

from utils.common_utils import regression_train, regression_test
from utils.model_utils import get_model
from easydict import EasyDict as edict
from utils.utils import restore_model
from utils.logging import get_logger, makedirs
from utils.al_utils import fetch_active_learning_dataset, get_selected_data_idxs, update_dataset
from global_settings.constants import RESULT_AL_PATH, LOG_AL_SELECTION_PATH, RESULT_AL_SELECTION_PATH, ORACLE_CKPT_DIR

def _run(args, seed, print):
    MODEL_PATH = osp.join(ORACLE_CKPT_DIR, args.dataset, 'seed%d' % seed)

    if not osp.exists(osp.join(RESULT_AL_SELECTION_PATH, args.criteria, '%s_seed%d' % (args.dataset, seed))):
        makedirs(osp.join(RESULT_AL_SELECTION_PATH, args.criteria, '%s_seed%d' % (args.dataset, seed), 'a.py'))
    raw_dataset = fetch_active_learning_dataset(args, seed)
    np.save(osp.join(RESULT_AL_SELECTION_PATH,
                     args.criteria,
                     '%s_seed%d' % (args.dataset, seed),
                     '%s_itr%d.npy' % ("oracle", 0)),
            {'test_x': raw_dataset.test_x,
             'test_y': raw_dataset.test_y,
             'train_x': raw_dataset.train_x,
             'train_y': raw_dataset.train_y,
             'pool_x': raw_dataset.pool_x,
             'pool_y': raw_dataset.pool_y})
    y_std = np.std(raw_dataset.train_y, 0, keepdims=True)
    y_std[y_std == 0] = 1
    y_mean = np.mean(raw_dataset.train_y, 0, keepdims=True)

    test_rmses, test_llds = [], []
    BSZ, NBS = args.batch_size, args.n_base
    for i_point in range(1, args.active_iterations + 2):
        tf.reset_default_graph()
        train_x, test_x, pool_x = raw_dataset.train_x, raw_dataset.test_x, raw_dataset.pool_x
        train_y, test_y, pool_y = raw_dataset.train_y, raw_dataset.test_y, raw_dataset.pool_y
        train_y, test_y, pool_y = (train_y - y_mean) / y_std, (test_y - y_mean) / y_std, (pool_y - y_mean) / y_std
        optimizer_dataset = edict({
            'train_x': train_x, 'train_y': train_y,
            'test_x': test_x, 'test_y': test_y,
            'pool_x': pool_x, 'pool_y': pool_y,
            'std_y_train': y_std,
        })
        N, input_dim = train_x.shape
        args.batch_size, args.n_base = min(BSZ, N), min(NBS, BSZ, N-1)

        oracle_n = tf.Variable(N, trainable=False)
        model, print_values, train_op, obs_var, corr_op, covar_op = get_model(
            args, train_x, train_y, test_x, pool_x, input_dim, print, oracle_N=oracle_n)
        global_step = model.global_step
        train_summary = tf.no_op()
        saver = tf.train.Saver(max_to_keep=1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        restore_model(args, print, saver, sess, MODEL_PATH)
        if args.method == 'svgp':
            retrain_vars = tf.trainable_variables(model.GP.name)
            sess.run(tf.variables_initializer(retrain_vars))
            sess.run(tf.assign(oracle_n, N))

        test_rmse, test_lld = None, None
        if args.method == 'svgp':
            for epoch in range(1, args.epochs+1):
                regression_train(args, model, model.infer_only_Z, optimizer_dataset, epoch, print_values,
                                 global_step, train_summary, None, sess, logger=print)
                if epoch % args.test_interval == 0 or epoch == args.epochs:
                    test_rmse, test_lld = regression_test(args, model, optimizer_dataset, epoch, sess, obs_var, global_step,
                                                          None, seed, logger=print)
        elif args.method == 'gp':
            test_rmse, test_lld = regression_test(args, model, optimizer_dataset, 0, sess, obs_var, global_step,
                                                  None, seed, logger=print)
        else:
            raise NotImplementedError

        test_rmses.append(test_rmse[0])
        test_llds.append(test_lld[0])

        # ================== select the new data point(s). ===========================================
        data_idxs = get_selected_data_idxs(train_x, test_x, pool_x, optimizer_dataset, args, covar_op, corr_op,
                                           model, sess)
        raw_dataset = update_dataset(raw_dataset, data_idxs)
        np.save(osp.join(RESULT_AL_SELECTION_PATH,
                         args.criteria,
                         '%s_seed%d' % (args.dataset, seed),
                         '%s_itr%d.npy' % ("oracle", i_point)),
                {'test_x': raw_dataset.test_x,
                 'test_y': raw_dataset.test_y,
                 'train_x': raw_dataset.train_x,
                 'train_y': raw_dataset.train_y})

        print("Iteration %d done. Selected data point %d/%d. (%d/%d)" % (i_point,
                                                                         i_point,
                                                                         args.active_iterations,
                                                                         raw_dataset.train_x.shape[0],
                                                                         raw_dataset.test_x.shape[0]))

    return test_rmses, test_llds


def run(args):
    log_dir = osp.join(LOG_AL_SELECTION_PATH, '%s_%s'%(args.dataset, 'oracle'))
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

    result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json'%(args.dataset, 'oracle', 'oracle'))
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
