
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import tensorflow as tf
import numpy as np
import core.gpflowSlim as gfs
from easydict import EasyDict as edict

from utils.logging import get_logger
from utils.common_utils import (regression_train,
                                regression_test)
from utils.model_utils import get_model
from utils.utils import save_model
from utils.al_utils import normalize_active_learning_data, fetch_active_learning_dataset
from global_settings.constants import LOG_AL_PRETRAIN_ORACLE_PATH, ORACLE_CKPT_DIR


def _run(args, seed, print):
    tf.reset_default_graph()
    MODEL_PATH = osp.join(ORACLE_CKPT_DIR, args.dataset, 'seed%d' % seed)

    ######## merge train+pool for pretraining the ORACLE. #########
    dataset = fetch_active_learning_dataset(args, seed)
    dataset = normalize_active_learning_data(dataset, has_pool=True)
    dataset = edict({
        'train_x': np.concatenate([dataset.train_x, dataset.pool_x], 0),
        'train_y': np.concatenate([dataset.train_y, dataset.pool_y], 0),
        'test_x': dataset.test_x,
        'test_y': dataset.test_y,
        'mean_y_train': dataset.mean_y_train,
        'std_y_train': dataset.std_y_train,
    })
    N, input_dim = dataset.train_x.shape

    ############ setup the MODEL ##############
    oracle_n = tf.Variable(N, trainable=False)
    model, print_values, train_op, obs_var, corr_op, covar_op = get_model(
        args, dataset.train_x, dataset.train_y, None, None, input_dim, print, oracle_N=oracle_n)
    global_step = model.global_step
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_summary = tf.no_op()
    saver = tf.train.Saver(max_to_keep=1)

    for epoch in range(1, args.epochs + 1):
        regression_train(args, model, train_op, dataset, epoch, print_values, global_step,
                         train_summary, None, sess, logger=print)
        if epoch % args.test_interval == 0 or epoch == args.epochs:
            regression_test(args, model, dataset, epoch, sess, obs_var, global_step, None, seed, logger=print)
            save_model(args, print, saver, sess, MODEL_PATH, epoch)


def run(args):
    log_dir = osp.join(LOG_AL_PRETRAIN_ORACLE_PATH, args.dataset)
    logger = get_logger(args.dataset, log_dir, __file__)
    print = logger.info

    # set jitter
    if args.dataset in ['concrete']:
        gfs.settings.set_jitter(3e-5)
    if args.dataset in ['naval']:
        gfs.settings.set_jitter(1e-4)

    for seed in range(args.init_seed, args.n_runs+args.init_seed):
        _run(args, seed, print)
