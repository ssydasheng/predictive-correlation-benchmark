import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
import core.gpflowSlim as gfs
import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split

from utils.logging import get_logger, makedirs
from data import uci_woval
from utils.model_utils import get_model
from utils.common_utils import regression_train
from global_settings.constants import RESULT_REG_PATH, LOG_REG_PATH
from utils.al_utils import fetch_active_learning_dataset, normalize_active_learning_data

def _run(args, seed, lprint):
    # init graph
    tf.reset_default_graph()
    summary_writer = None

    dataset = fetch_active_learning_dataset(args, seed)
    dataset = normalize_active_learning_data(dataset)
    train_x, test_x, valid_x = dataset.train_x, dataset.test_x, dataset.pool_x
    train_y, test_y, valid_y = dataset.train_y, dataset.test_y, dataset.pool_y
    std_y_train = dataset.std_y_train[0]

    N, input_dim = train_x.shape
    args.batch_size, args.n_base = min(args.batch_size, N), min(args.n_base, args.batch_size, N - 1)
    # setup model
    model, print_values, train_op, obs_var, _, _ = get_model(args, train_x, train_y, test_x, None, input_dim, lprint)
    test_obs_var = obs_var

    # setup summary
    train_summary = tf.no_op()
    global_step = model.global_step

    # training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    best_epoch, best_rmse, best_lld = 0, np.inf, -np.inf  # for validation
    best_test_rmse, best_test_lld = np.inf, -np.inf  # for test
    learning_rate = args.learning_rate

    begin_epoch = 1
    for epoch in range(begin_epoch, args.epochs + 1):
        if epoch % args.lr_ann_epochs == 0:
            learning_rate = learning_rate * args.lr_ann_ratio
        regression_train(args, model, train_op, dataset, epoch, print_values, global_step,
                         train_summary, summary_writer, sess, learning_rate, lprint)

        # ================================== valid ==================================
        if (epoch % args.test_interval == 0 or epoch == args.epochs):
            feed_dict = {model.x: train_x,
                         model.y: train_y,
                         model.x_pred: valid_x,
                         model.y_pred: valid_y,
                         model.n_particles: args.test_samples}
            rmse, lld, ov, gs = sess.run([model.eval_rmse, model.eval_lld, obs_var, global_step], feed_dict=feed_dict)
            rmse, lld = rmse * std_y_train, lld - np.log(std_y_train)

            lprint('>>> VALID: Seed {:5d} >>> Epoch {:5d}/{:5d} | rmse={:.5f} | lld={:.5f} | obs_var={:.5f}'.format(
                seed, epoch, args.epochs, rmse, lld, np.mean(ov)))
            if lld > best_lld:
                best_epoch, best_rmse, best_lld = epoch, rmse, lld

        # ================================== testing ==================================
        if epoch % args.test_interval == 0 or epoch == args.epochs:
            feed_dict = {model.x_pred: test_x,
                         model.y_pred: test_y,
                         model.x: train_x,
                         model.y: train_y,
                         model.n_particles: args.test_samples}
            rmse, lld, ov, gs = sess.run([model.eval_rmse, model.eval_lld, test_obs_var, global_step], feed_dict=feed_dict)
            rmse, lld = rmse * std_y_train, lld - np.log(std_y_train)
            lprint('>>> TEST: Seed {:5d} >>> Epoch {:5d}/{:5d} | rmse={:.5f} | lld={:.5f} | obs_var={:.5f}'.format(
                    seed, epoch, args.epochs, rmse, lld, np.mean(ov)))
            if best_epoch == epoch:
                # save mean and covariance for computing XLL(R)
                mean, cov = sess.run([model.func_x_pred_mean, model.y_x_pred_cov], feed_dict=feed_dict)
                method = 'gp' if args.method == 'svgp' else args.method
                result_dir = osp.join(RESULT_REG_PATH, '%s_%s' % (args.dataset, method))
                if not osp.exists(result_dir):
                    makedirs(osp.join(result_dir, 'a.py'))
                with open(osp.join(result_dir, 'test_mean_cov_seed%d.npz' % seed), 'wb') as file:
                    np.savez(file, mean=mean, cov=cov)

                best_test_lld, best_test_rmse = lld, rmse
                lprint('BEST EPOCH !!!')
            if epoch == args.epochs:
                if args.return_val:
                    return best_rmse, best_lld, best_test_rmse, best_test_lld
                else:
                    return best_rmse, best_lld, best_test_rmse, best_test_lld


def run(args):
    method = 'gp' if args.method == 'svgp' else args.method
    log_dir = osp.join(LOG_REG_PATH, '%s_%s'%(args.dataset, method))
    # setup logger
    logger = get_logger(args.dataset, log_dir, __file__)
    print = logger.info

    # set jitter
    if args.dataset in ['concrete']:
        gfs.settings.set_jitter(3e-5)
    if args.dataset in ['naval']:
        gfs.settings.set_jitter(1e-4)

    rmse_results, lld_results = [], []
    test_rmse_results, test_lld_results = [], []

    for seed in range(args.init_seed, args.n_runs + args.init_seed):
        rmse, ll, test_rmse, test_lld = _run(args, seed, print)
        rmse_results.append(rmse)
        lld_results.append(ll)
        test_rmse_results.append(test_rmse)
        test_lld_results.append(test_lld)

    print('test rmse = {}'.format(rmse_results))
    print('test log likelihood = {}'.format(lld_results))
    print("Test rmse =                   {}/{}".format(np.mean(rmse_results),
                                                       np.std(rmse_results) / args.n_runs ** 0.5))
    print("Test log likelihood =         {}/{}".format(np.mean(lld_results),
                                                       np.std(lld_results) / args.n_runs ** 0.5))
    print('NOTE: Test result above output mean and std. errors')

    result_dir = osp.join(RESULT_REG_PATH, '%s_%s'%(args.dataset, method))
    with open(osp.join(result_dir, 'res.json'), 'w') as f:
        res = {
            'rmse': [np.mean(rmse_results), np.std(rmse_results) / args.n_runs ** 0.5],
            'lld': [np.mean(lld_results), np.std(lld_results) / args.n_runs ** 0.5],
            'test_rmse': [np.mean(test_rmse_results), np.std(test_rmse_results) / args.n_runs ** 0.5],
            'test_lld': [np.mean(test_lld_results), np.std(test_lld_results) / args.n_runs ** 0.5],
            'all_rmse': rmse_results,
            'all_lld': lld_results
        }
        json.dump(res, f)
