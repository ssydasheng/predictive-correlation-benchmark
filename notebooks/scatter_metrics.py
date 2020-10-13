import sys
import os
import os.path as osp
root_path = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(root_path)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set(style="whitegrid")

import numpy as np
from scipy import stats
import tensorflow as tf
from easydict import EasyDict

from global_settings.constants import RESULT_TOY_METRIC_PATH, TOY_DATA_PATH
from utils.xll_utils import get_indices, get_submatrices, build_graph


def init_plotting(legend_ratio):
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 1.7 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.7 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = legend_ratio * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.2 * plt.rcParams['font.size']


BASE_PATH = osp.join(root_path, RESULT_TOY_METRIC_PATH)
N_SUB=5
N_SAMPLES=5000
INDICES_TYPE = 'topk_abs'

colors = {
    'BBB': '#e874b9',
    'Dropout': '#b15928',
    'HMC': '#6a3d9a',
    'Ensemble': '#33a02c',
    'GP-RBF': '#737373',
    'FBNN': '#e31a1c',
    'NNG': '#1f78b4'
}

def load_reference(seed, reference):
    name, VAR, LR, EP, NU = reference
    path = os.path.join(RESULT_TOY_METRIC_PATH, '{}_seed{}_VAR{}_NU{}_LR{}_EP{}.npz'.format(name, seed, VAR, NU, LR, EP))
    res = np.load(path)
    mean, cov = res['test_pred_mean'], res['test_pred_cov']
    return mean, cov

def load_method(y_test, refer_mean, refer_cov, seed, dataset_name, method, log_prob_tf, mean_tf, y_tf, cov_tf, sess, n_sub):
    name, VAR, LR, EP, NU = method
    path = os.path.join(RESULT_TOY_METRIC_PATH,
                        '{}_seed{}_VAR{}_NU{}_LR{}_EP{}.npz'.format(name, seed, VAR, NU, LR, EP))
    res = np.load(path)
    mean, cov = res['test_pred_mean'], res['test_pred_cov']

    np.random.seed(seed)
    indices_es = get_indices(INDICES_TYPE, len(y_test), n_sub, N_SAMPLES, refer_cov)

    y_test_pair = np.concatenate([y_test[ind][:, None] for ind in indices_es], axis=1) # [N, 2]
    cov_pair = get_submatrices(cov, *indices_es) # [N, 2, 2]

    refer_mean_pair = np.concatenate([refer_mean[ind][:, None] for ind in indices_es], axis=1) # [N, 2]
    refer_cov_pair = get_submatrices(refer_cov, *indices_es) # [N, 2, 2]
    diag_ratio = (np.diagonal(refer_cov_pair, axis1=1, axis2=2) / np.diagonal(cov_pair, axis1=1, axis2=2))[..., None]**0.5
    rescaled_cov_pair = cov_pair * diag_ratio * np.transpose(diag_ratio, [0, 2, 1])
    p2_lld = sess.run(log_prob_tf, feed_dict={mean_tf: refer_mean_pair, y_tf: y_test_pair, cov_tf: rescaled_cov_pair}) / n_sub

    return p2_lld


def load_refer_dataset(METHODS, dataset_name, reference, log_prob_tf, mean_tf, y_tf, cov_tf, sess):
    res_p2_lld = {method: [] for method in METHODS}
    for seed in range(1, 51):
        dataset = EasyDict(dict(np.load(os.path.join(
            root_path, TOY_DATA_PATH, 'reluLarge', 'dim%d_seed%s.npz' % (9, seed)))))
        refer_mean, refer_cov = load_reference(seed, reference)
        for method in METHODS:
            p2_lld = load_method(
                dataset.test_y, refer_mean, refer_cov, seed, dataset_name, method, log_prob_tf, mean_tf, y_tf, cov_tf, sess, N_SUB)
            y_std = 1 if not hasattr(dataset, 'std_y_train') else dataset.std_y_train
            res_p2_lld[method].append(p2_lld - np.log(y_std))
    for method in METHODS:
        print('REFERENCE {} | {} | {:10s} | {} | p2_lld={:4f} '.format(
            reference, N_SUB, dataset_name, method, np.mean(res_p2_lld[method])))
    print('------------------------------------------------------------')
    return {method: [float(v) for v in value] for method, value in res_p2_lld.items()}


##################### compute the marginal llds for method #####################
def load_method_marginal(method):
    llds = []
    for seed in range(1, 51):
        dataset = EasyDict(dict(np.load(os.path.join(
            root_path, TOY_DATA_PATH, 'reluLarge', 'dim%d_seed%s.npz' % (9, seed)))))
        name, VAR, LR, EP, NU = method
        path = os.path.join(RESULT_TOY_METRIC_PATH,
                            '{}_seed{}_VAR{}_NU{}_LR{}_EP{}.npz'.format(name, seed, VAR, NU, LR, EP))
        if not osp.exists(path):
            return None
        res = np.load(path)
        mean, cov = res['test_pred_mean'], res['test_pred_cov']
        m1_lld = np.mean(np.log(stats.norm.pdf(dataset.test_y, loc=mean, scale=np.diagonal(cov) ** 0.5)))
        y_std = 1 if not hasattr(dataset, 'std_y_train') else dataset.std_y_train
        llds.append(m1_lld - np.log(y_std))
    return np.mean(llds)


##################### get all available methods #####################
###### if top_k: filter out all methods with good marginal llds #####
def filter_methods(top_k):
    METHODS, METHODS_NAMES, LLDs = [], [], []
    methods = ['bbb', 'dropout', 'emhmc', 'homo_ensemble', 'gp', 'fbnn', 'nng']
    method_names = ['BBB', 'Dropout', 'HMC', 'Ensemble', 'GP-RBF', 'FBNN', 'NNG']

    for idx, name in enumerate(methods):
        vars = [1] if name in ['gp', 'fbnn'] else [3, 6, 9, 10, 20, 50, 100, 150, 200]
        for VAR in vars:
            for LR in [0.001, 0.003, 0.01]:
                for EP in [100, 300, 1000]:
                    NUs = [1] if name == 'gp' else [50, 100]
                    for NU in NUs:
                        lld = load_method_marginal((name, VAR, LR, EP, NU))
                        if lld is not None:
                            METHODS.append((name, VAR, LR, EP, NU))
                            METHODS_NAMES.append(method_names[idx])
                            LLDs.append(lld)

    if top_k:
        LLDs = np.asarray(LLDs)
        num_per_model = len([a for a in METHODS_NAMES if a == 'GP-RBF'])
        new_METHODS, new_METHODS_NAMES = [], []
        for name in method_names:
            top_hps = np.array([METHODS_NAMES[idx]==name for idx in range(len(LLDs))])
            locs = np.where(top_hps, LLDs, np.ones_like(LLDs)*(-np.inf))
            top_hps = np.argsort(locs)[-num_per_model:]
            new_METHODS.extend([METHODS[idx] for idx in top_hps])
            new_METHODS_NAMES.extend([METHODS_NAMES[idx] for idx in top_hps])
        METHODS, METHODS_NAMES = new_METHODS, new_METHODS_NAMES
    return METHODS, METHODS_NAMES


##################### load TAL llds, meta-corrs for METHODS #####################
def load_metrics(METHODS):
    mean_corr_prs_f, mean_corr_sprs_f = [], []
    mean_var_prs_f, mean_var_sprs_f = [], []
    mean_TIG_test_llds, mean_BMIG_test_llds = [], []
    for method in METHODS:
        corr_prs_f, corr_sprs_f = [], []
        var_prs_f, var_sprs_f = [], []
        TIG_test_llds, BMIG_test_llds = [], []
        for seed in range(1, 51):
            name, VAR, LR, EP, NU = method
            path = os.path.join(RESULT_TOY_METRIC_PATH,
                                '{}_seed{}_VAR{}_NU{}_LR{}_EP{}.npz'.format(name, seed, VAR, NU, LR, EP))
            res = np.load(path)
            corr_prs_f.append(res['corr_pr_f'])
            corr_sprs_f.append(res['corr_spr_f'])

            var_prs_f.append(res['var_pr_f'])
            var_sprs_f.append(res['var_spr_f'])

            TIG_test_llds.append(res['TIG_test_lld'])
            BMIG_test_llds.append(res['BMIG_test_lld'])

        mean_corr_prs_f.append(np.mean(corr_prs_f))
        mean_corr_sprs_f.append(np.mean(corr_sprs_f))

        mean_var_prs_f.append(np.mean(var_prs_f))
        mean_var_sprs_f.append(np.mean(var_sprs_f))

        mean_TIG_test_llds.append(np.mean(TIG_test_llds))
        mean_BMIG_test_llds.append(np.mean(BMIG_test_llds))
    return np.asarray(mean_corr_prs_f), np.asarray(mean_corr_sprs_f), \
           np.asarray(mean_var_prs_f), np.asarray(mean_var_sprs_f), \
           np.asarray(mean_TIG_test_llds), np.asarray(mean_BMIG_test_llds)


############## scatterplot, comparing TAL performances vs meta-corr ##############
def scatter_TAL_meta_corr():
    print('Start TAL -- corr')

    METHODS, METHODS_NAMES = filter_methods(top_k=False)
    mean_corr_prs_f, _, mean_var_prs_f, _, mean_TIG_test_llds, mean_BMIG_test_llds = load_metrics(METHODS)

    print('Finish Loading')
    init_plotting(1.5)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) # sharey=True
    for corr, name, lld in zip(mean_corr_prs_f, METHODS_NAMES, mean_BMIG_test_llds):
        ax[0].scatter(corr, lld, color=colors[name], s=32)

    ax[0].set_xlabel('Pearson Correlation of \n Correlations')
    ax[0].set_ylabel('LLD (BatchMIG)')

    flags = {a: True for a in colors.keys()}
    for var, name, lld in zip(mean_var_prs_f, METHODS_NAMES, mean_TIG_test_llds):
        if flags[name]:
            ax[1].scatter(var, lld, color=colors[name], s=32, label=name)
        else:
            ax[1].scatter(var, lld, color=colors[name], s=32)
        flags[name] = False
    ax[1].set_xlabel('Pearson Correlation of \n Variances')
    ax[1].set_ylabel('LLD (TIG)')
    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.52, 1.02),
                     fancybox=False, ncol=4, markerscale=2)
    plt.tight_layout(rect=(0,0,1,0.85))
    plt.savefig('notebooks/figures/scatter_TAL_meta_corr.pdf')


############## compute the xll(r) for METHODS ##############
def load_xll(METHODS):
    dataset_name = ''
    log_prob_tf, mean_tf, y_tf, cov_tf = build_graph(N_SUB)
    sess = tf.Session()

    mean_results = {method: [] for method in METHODS}
    rank_results = {method: [] for method in METHODS}
    for idx, reference in enumerate(METHODS):
        print(idx)
        print(reference)
        res = load_refer_dataset(METHODS, dataset_name, reference, log_prob_tf, mean_tf, y_tf, cov_tf, sess)
        for method in METHODS:
            mean_results[method].extend(res[method])
        names, values = zip(*res.items())
        names, values = list(names), np.array(list(values))
        for seed in range(1, 51):
            order = np.argsort(values[:, seed-1])[::-1]
            for idx in range(len(names)):
                rank_results[names[order[idx]]].append(idx)

    for method in METHODS:
        mean_results[method] = np.mean(mean_results[method])
        rank_results[method] = np.mean(rank_results[method])

    mean_results = [mean_results[method] for method in METHODS]
    rank_results = [rank_results[method] for method in METHODS]
    return np.asarray(mean_results), np.asarray(rank_results)


############## scatterplot, comparing TAL performances , meta-corr, xll ##############
def scatter_xll():
    print('Start XLL')
    METHODS, METHODS_NAMES = filter_methods(top_k=True)
    corr_prs_f, corr_sprs_f, _, _, _, BMIG_test_llds = load_metrics(METHODS)
    xll, xllr = load_xll(METHODS)

    np.savez('notebooks/figures/xll.npz',
            xll=xll, xllr=xllr,
            corr_prs_f=corr_prs_f, corr_sprs_f=corr_sprs_f,
            BMIG_test_llds=BMIG_test_llds)

    print('Finish Loading')

    init_plotting(1.8)
    scatter_colors = [colors[key] for key in METHODS_NAMES]

    plt.rcParams['legend.fontsize'] = 1.4 * plt.rcParams['font.size']
    plt.rcParams['axes.labelsize'] = 1.9 * plt.rcParams['font.size']

    fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex='col', sharey='row')

    ax[0, 0].scatter(xll, corr_prs_f, color=scatter_colors, s=50)
    ax[0, 0].set_ylabel('meta-correlations')

    labels = [name for idx, name in enumerate(METHODS_NAMES) if (name not in METHODS_NAMES[:idx])]
    for name in labels:
        flag = np.asarray([idx for idx in range(len(METHODS_NAMES)) if name == METHODS_NAMES[idx]])
        ax[1, 0].scatter(xll[flag], BMIG_test_llds[flag], color=colors[name], s=50)
    ax[1, 0].set_ylabel('LLD (BatchMIG)')
    ax[1, 0].set_xlabel('XLL')

    ax[0, 1].scatter(xllr, corr_prs_f, color=scatter_colors, s=50)
    labels = [name for idx, name in enumerate(METHODS_NAMES) if (name not in METHODS_NAMES[:idx])]
    for name in labels:
        flag = np.asarray([idx for idx in range(len(METHODS_NAMES)) if name == METHODS_NAMES[idx]])
        ax[1, 1].scatter(xllr[flag], BMIG_test_llds[flag], color=colors[name], s=50, label=name)
    ax[1, 1].set_xlabel('XLLR')

    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.52, 0.99),
                   fancybox=False, ncol=4, markerscale=2)
    plt.tight_layout(rect=(0,0,1,0.9))
    fig.align_ylabels(ax)
    plt.savefig('notebooks/figures/scatter_xll.pdf')


if __name__ == '__main__':
    scatter_TAL_meta_corr()
    scatter_xll()