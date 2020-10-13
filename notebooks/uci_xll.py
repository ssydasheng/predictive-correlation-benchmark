import sys
import os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import tensorflow as tf

from data import uci_woval
from utils.al_utils import HParams, fetch_active_learning_dataset, normalize_active_learning_data
from utils.xll_utils import get_indices, get_submatrices, build_graph
from global_settings.constants import RESULT_REG_PATH

N_SUB=5
N_SAMPLES=5000
INDICES_TYPE = 'topk_abs'
# METHODS = ['bbb', 'ensemble', 'dropout', 'nng', 'emhmc', 'gp', 'fbnn']
METHODS = ['bbb', 'ensemble', 'dropout', 'nng', 'emhmc', 'gp', 'fbnn']

def load_reference(seed, dataset_name, reference):
    path = osp.join(RESULT_REG_PATH, '%s_%s' % (dataset_name, reference), 'test_mean_cov_seed%d.npz' % seed)
    res = np.load(path)
    mean, cov = res['mean'], res['cov']
    return mean, cov

def load_method(y_test, refer_mean, refer_cov, seed, dataset_name, method, log_prob_tf, mean_tf, y_tf, cov_tf, sess, n_sub):
    path = osp.join(RESULT_REG_PATH, '%s_%s' % (dataset_name, method), 'test_mean_cov_seed%d.npz' % seed)
    res = np.load(path)
    mean, cov = res['mean'], res['cov']

    rmse = np.sqrt(np.mean((mean - y_test)**2.))
    m1_lld = np.mean(np.log(stats.norm.pdf(y_test, loc=mean, scale=np.diagonal(cov) ** 0.5)))

    np.random.seed(seed)
    indices_es = get_indices(INDICES_TYPE, len(y_test), n_sub, N_SAMPLES, refer_cov)

    y_test_part = np.concatenate([y_test[ind] for ind in indices_es], axis=0)
    mean_part = np.concatenate([mean[ind] for ind in indices_es], axis=0)
    var_part = np.concatenate([np.diagonal(cov)[ind] for ind in indices_es], axis=0)
    m2_lld = np.mean(np.log(stats.norm.pdf(y_test_part, loc=mean_part, scale=var_part ** 0.5)))

    y_test_pair = np.concatenate([y_test[ind][:, None] for ind in indices_es], axis=1) # [N, 2]
    mean_pair = np.concatenate([mean[ind][:, None] for ind in indices_es], axis=1) # [N, 2]
    cov_pair = get_submatrices(cov, *indices_es) # [N, 2, 2]
    p1_lld = sess.run(log_prob_tf, feed_dict={mean_tf: mean_pair, y_tf: y_test_pair, cov_tf: cov_pair}) / n_sub

    refer_mean_pair = np.concatenate([refer_mean[ind][:, None] for ind in indices_es], axis=1) # [N, 2]
    refer_cov_pair = get_submatrices(refer_cov, *indices_es) # [N, 2, 2]
    diag_ratio = (np.diagonal(refer_cov_pair, axis1=1, axis2=2) / np.diagonal(cov_pair, axis1=1, axis2=2))[..., None]**0.5
    rescaled_cov_pair = cov_pair * diag_ratio * np.transpose(diag_ratio, [0, 2, 1])
    p2_lld = sess.run(log_prob_tf, feed_dict={mean_tf: refer_mean_pair, y_tf: y_test_pair, cov_tf: rescaled_cov_pair}) / n_sub

    return rmse, m1_lld, m2_lld, p1_lld, p2_lld

def load_refer_dataset(dataset_name, reference, log_prob_tf, mean_tf, y_tf, cov_tf, sess):
    res_rmse = {method: [] for method in METHODS}
    res_m1_lld = {method: [] for method in METHODS}
    res_m2_lld = {method: [] for method in METHODS}
    res_p1_lld = {method: [] for method in METHODS}
    res_p2_lld = {method: [] for method in METHODS}
    for seed in range(1, 11):
        dataset = fetch_active_learning_dataset(
            HParams(test_ratio=0.2, train_ratio=0.2, dataset=dataset_name), seed)
        dataset = normalize_active_learning_data(dataset)
        refer_mean, refer_cov = load_reference(seed, dataset_name, reference)
        for method in METHODS:
            rmse, m1_lld, m2_lld, p1_lld, p2_lld = load_method(
                dataset.test_y, refer_mean, refer_cov, seed, dataset_name, method, log_prob_tf, mean_tf, y_tf, cov_tf, sess, N_SUB)
            res_rmse[method].append(rmse * dataset.std_y_train)
            res_m1_lld[method].append(m1_lld - np.log(dataset.std_y_train))
            res_m2_lld[method].append(m2_lld - np.log(dataset.std_y_train))
            res_p1_lld[method].append(p1_lld - np.log(dataset.std_y_train))
            res_p2_lld[method].append(p2_lld - np.log(dataset.std_y_train))
    for method in METHODS:
        print('REFERENCE {:8s} | RT {:.2f} | {} | {:10s} | {:10s} | rmse={:4f} | m1_lld={:4f} | p1_lld={:4f} | p2_lld={:4f} '.format(
            reference, 0.2, N_SUB, dataset_name, method, np.mean(res_rmse[method]), np.mean(res_m1_lld[method]), np.mean(res_p1_lld[method]), np.mean(res_p2_lld[method])))
    print('------------------------------------------------------------')
    return {method: [float(v) for v in value] for method, value in res_p2_lld.items()}

#####################################################################################################
# This function uses every method as a reference, and compute the average lld/rank for all methods.
def load_dataset(dataset_name):
    log_prob_tf, mean_tf, y_tf, cov_tf = build_graph(N_SUB)
    sess = tf.Session()

    mean_results = {method: [] for method in METHODS}
    rank_results = {method: [] for method in METHODS}
    for reference in METHODS:
        print(reference)
        res = load_refer_dataset(dataset_name, reference, log_prob_tf, mean_tf, y_tf, cov_tf, sess)
        for method in METHODS:
            mean_results[method].extend(res[method])
        names, values = zip(*res.items())
        names, values = list(names), np.array(list(values))
        for seed in range(1, 11):
            order = np.argsort(values[:, seed-1])[::-1]
            for idx in range(len(names)):
                rank_results[names[order[idx]]].append(idx)

    for method in METHODS:
        print('{:10s} | RT {:.2f} | {:10s} | mean_lld={:4f} ({:4f}) '.format(
            dataset_name, 0.2, method, np.mean(mean_results[method]),
            np.std(mean_results[method])/len(mean_results[method])**0.5))
    for method in METHODS:
        print('{:10s} | RT {:.2f} | {:10s} | mean_rank={:4f} ({:4f}) '.format(
            dataset_name, 0.2, method, np.mean(rank_results[method]),
            np.std(rank_results[method])/len(rank_results[method])**0.5))

    for method in ['gp', 'bbb', 'nng', 'emhmc', 'fbnn', 'dropout', 'ensemble']:
        print('&%.3f (%.3f)' % (np.mean(mean_results[method]), np.std(mean_results[method])/len(mean_results[method])**0.5))
    for method in ['gp', 'bbb', 'nng', 'emhmc', 'fbnn', 'dropout', 'ensemble']:
        print('&%.2f (%.2f)' % (np.mean(rank_results[method]), np.std(rank_results[method])/len(rank_results[method])**0.5))

if __name__ == '__main__':

    a = load_dataset('boston')
    b = load_dataset('concrete')
    c = load_dataset('energy')
    d = load_dataset('wine')
    e = load_dataset('yacht')
    f = load_dataset('kin8nm')
    g = load_dataset('naval')
    h = load_dataset('power_plant')
